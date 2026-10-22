"""
A small RFC 6455 websocket, standard library only (the venv has no
websockets package and installing one is not needed for this): the
server-side handshake and framing used by app/server.py, and a blocking
client for tests, scripts and the ROS 2 bridge.

Text and binary messages, fragmentation (continuation frames), ping/pong
and close; client frames are masked as the RFC requires, server frames are
not. A peer that breaks the RFC gets ProtocolError (the server closes the
connection with its status code): unmasked client frames, reserved bits or
opcodes, fragmented or long control frames, a continuation without a start,
invalid UTF-8 in a text message (1007), a message whose fragments together
exceed MAX_MSG (1009). No extensions (no permessage-deflate) and no TLS --
this serves 127.0.0.1.
"""
import base64
import hashlib
import json
import os
import socket
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0, 1, 2, 8, 9, 10
MAX_MSG = 4 * 2 ** 20          # one message, all its fragments


def accept_key(key):
    return base64.b64encode(hashlib.sha1((key + GUID).encode())
                            .digest()).decode()


def _read_exact(f, n):
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise ConnectionError("websocket closed")
        buf += chunk
    return buf


class ProtocolError(Exception):
    """A peer broke RFC 6455: close the connection with `code` (1002
    protocol error, 1007 invalid payload, 1009 too big)."""

    def __init__(self, code, reason):
        super().__init__(f"{code}: {reason}")
        self.code, self.reason = int(code), str(reason)


def read_frame(f, require_mask=None):
    """(fin, opcode, payload) of one frame from a binary file object.
    require_mask True (a server reading a client): unmasked frames are a
    protocol error (RFC 6455 5.1); False (a client reading a server):
    masked frames are."""
    b0, b1 = _read_exact(f, 2)
    fin, op = bool(b0 & 0x80), b0 & 0x0F
    masked, n = bool(b1 & 0x80), b1 & 0x7F
    if b0 & 0x70:
        raise ProtocolError(1002, "reserved bits set (no extensions)")
    if op not in (OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG):
        raise ProtocolError(1002, f"unknown opcode {op}")
    if require_mask is not None and masked != require_mask:
        raise ProtocolError(1002, "client frames must be masked"
                            if require_mask else "server frames must not "
                            "be masked")
    if n == 126:
        n = struct.unpack(">H", _read_exact(f, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", _read_exact(f, 8))[0]
    if op >= OP_CLOSE and (n > 125 or not fin):
        raise ProtocolError(1002, "control frames: unfragmented, <= 125 "
                            "bytes")
    if n > MAX_MSG:
        raise ProtocolError(1009, f"frame of {n} bytes refused (max "
                            f"{MAX_MSG})")
    mask = _read_exact(f, 4) if masked else None
    data = _read_exact(f, n)
    if mask:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    return fin, op, data


def write_frame(f, payload, opcode=OP_TEXT, mask=False):
    if isinstance(payload, str):
        payload = payload.encode()
    n = len(payload)
    head = bytes([0x80 | opcode])
    mbit = 0x80 if mask else 0
    if n < 126:
        head += bytes([mbit | n])
    elif n < 2 ** 16:
        head += bytes([mbit | 126]) + struct.pack(">H", n)
    else:
        head += bytes([mbit | 127]) + struct.pack(">Q", n)
    if mask:
        key = os.urandom(4)
        payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
        head += key
    f.write(head + payload)
    f.flush()


def read_message(f, write=None, require_mask=None):
    """One whole message (str for text, bytes for binary); None on close.
    Answers pings through write(payload, opcode) when given. Raises
    ProtocolError on a broken frame sequence, invalid UTF-8 in a text
    message, or a message (all its fragments) over MAX_MSG."""
    parts, first_op, total = [], None, 0
    while True:
        fin, op, data = read_frame(f, require_mask)
        if op == OP_CLOSE:
            if write is not None:
                try:
                    write(data[:2], OP_CLOSE)
                except OSError:
                    pass
            return None
        if op == OP_PING:
            if write is not None:
                write(data, OP_PONG)
            continue
        if op == OP_PONG:
            continue
        if op == OP_CONT:
            if first_op is None:
                raise ProtocolError(1002, "continuation without a first "
                                    "frame")
        else:
            if first_op is not None:
                raise ProtocolError(1002, "new message before the last "
                                    "one's final fragment")
            first_op = op
        total += len(data)
        if total > MAX_MSG:
            raise ProtocolError(1009, f"message over {MAX_MSG} bytes")
        parts.append(data)
        if fin:
            msg = b"".join(parts)
            if first_op == OP_TEXT:
                try:
                    return msg.decode("utf-8")
                except UnicodeDecodeError:
                    raise ProtocolError(1007, "text message is not valid "
                                        "UTF-8") from None
            return msg


class ServerSocket:
    """Server side over an http.server request handler that has done the
    handshake (handshake(handler))."""

    def __init__(self, rfile, wfile):
        self.rfile, self.wfile = rfile, wfile

    def send(self, payload, opcode=OP_TEXT):
        write_frame(self.wfile, payload, opcode, mask=False)

    def recv(self):
        return read_message(self.rfile, self.send, require_mask=True)

    def close(self, code=1000, reason=""):
        """Send a close frame (status code + reason); errors ignored."""
        try:
            write_frame(self.wfile, struct.pack(">H", int(code))
                        + str(reason).encode()[:120], OP_CLOSE, mask=False)
        except OSError:
            pass


def handshake(handler):
    """Answer an Upgrade request on a BaseHTTPRequestHandler; returns a
    ServerSocket or None (the handler has then sent an error)."""
    key = handler.headers.get("Sec-WebSocket-Key")
    if (handler.headers.get("Upgrade", "").lower() != "websocket"
            or not key):
        handler.send_error(400, "websocket upgrade expected")
        return None
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept_key(key))
    handler.end_headers()
    handler.wfile.flush()
    return ServerSocket(handler.rfile, handler.wfile)


class Client:
    """A blocking websocket client: Client(host, port, '/ws')."""

    def __init__(self, host="127.0.0.1", port=8766, path="/ws",
                 timeout=60.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.f = self.sock.makefile("rwb", buffering=0)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
               f"\r\n")
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            c = self.sock.recv(1)
            if not c:
                raise ConnectionError("no handshake reply")
            head += c
        lines = head.decode().split("\r\n")
        if " 101 " not in lines[0] + " ":
            raise ConnectionError(f"handshake refused: {lines[0]}")
        hdr = {ln.split(":", 1)[0].lower(): ln.split(":", 1)[1].strip()
               for ln in lines[1:] if ":" in ln}
        if hdr.get("sec-websocket-accept") != accept_key(key):
            raise ConnectionError("bad Sec-WebSocket-Accept")

    def send(self, payload, opcode=OP_TEXT):
        write_frame(self.f, payload, opcode, mask=True)

    def recv(self):
        return read_message(self.f, self.send, require_mask=False)

    def call(self, obj):
        """Send a JSON object, return the JSON reply."""
        self.send(json.dumps(obj))
        r = self.recv()
        return None if r is None else json.loads(r)

    def close(self):
        try:
            write_frame(self.f, struct.pack(">H", 1000), OP_CLOSE, mask=True)
        except OSError:
            pass
        self.sock.close()

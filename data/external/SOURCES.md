# External data: where each file came from

Downloaded for validation experiments (2026-09). Each folder: what it is, the source URL as
recorded when it was fetched, the licence as stated by the source, and what uses it. Hashes
are SHA-256 of the files as they sit here; a changed hash means a changed file.

Nothing here is redistributed by this project. Files without a stated licence are public
benchmark data used for comparison only.

## bseries/

- **What**: Wageningen B-series open-water polynomial coefficients (K_T, K_Q in P/D, AE/A0, Z, J).
- **Source**: https://raw.githubusercontent.com/mkergoat/bseries/master/coeffs.dat and .../LICENSE
- **Licence**: GPL-3.0 (LICENSE file of that repository)
- **Used by**: sim/propeller.py (WageningenB); studies/exp_propeller.py

| file | bytes | sha256 |
|---|---:|---|
| bseries/LICENSE | 35149 | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |
| bseries/coeffs.dat | 1782 | `aa8296e2d55130dcc8c7859cf791b05cdf6ff3481c70a8706ea4648302b87df2` |

## mss/

- **What**: Marine Systems Simulator (T. I. Fossen): Blendermann (1994) and Isherwood (1972) wind-load sets, the Otter USV model, Wageningen data.
- **Source**: https://github.com/cybergalactic/MSS (raw files from raw.githubusercontent.com/cybergalactic/MSS/master/...; WageningData.txt from LIBRARY/modeling/utiles/; tree.json = the repository listing from api.github.com)
- **Licence**: MIT, (c) 2004 Thor I. Fossen (LICENSE)
- **Used by**: sim/forces.py (BLENDERMANN table); cross-checks

| file | bytes | sha256 |
|---|---:|---|
| mss/LICENSE | 1071 | `922bb433450a9e39368009c29db7b07857520b8e3aad20797f5dc082df6dcce1` |
| mss/WageningData.txt | 4089 | `c8b827582b5db1254347ed4140bc04d2155451d7c7769b3961678878b2705ab7` |
| mss/blendermann94.m | 3286 | `896b803690164174e1492a183bfb01790616ec81599f27493ec0d0c44c3e00db` |
| mss/exWageningen.m | 2841 | `edac61621238e79f7362ab48d1f18f8c0e4b848eccae7bae9de42e03b6433a89` |
| mss/isherwood72.m | 6680 | `a0adacccb590bc7db4fbe27c8e06b9dbb9cc7158657b3295b2662c3ab5bf8961` |
| mss/otter.m | 9481 | `66c144fea629cee4bea0ad55b62a1bc5e55b46e68ac8e94860ddc7f0aff12248` |
| mss/tree.json | 150449 | `763c1140f424e65115274ca084816f94ce2dc69204ea87aeda2f92ca0157b0ec` |
| mss/wageningen.m | 1525 | `aba58038de38ecb205ed0cddc5fa88b9a3ccb9b8154daad919ef51fac6182bc0` |

## ikeda/

- **What**: Kawahara, Maekawa & Ikeda (2009), 'A simple prediction formula of roll damping of conventional cargo ships on the basis of Ikeda's method and its limitation', STAB 2009. The .txt is a text extraction. kawahara2009_fig1_markers.csv is DERIVED here: the Fig. 1 symbols read from the PDF's vector paths (see its header).
- **Source**: https://shipstab.org/files/Proceedings/STAB/STAB2009/STAB_2009_s07-p2.pdf
- **Licence**: (c) the authors / STAB proceedings; local reference copy, not for redistribution
- **Used by**: hydro/ikeda.py (formulas, verify)

| file | bytes | sha256 |
|---|---:|---|
| ikeda/Kawahara_2009_STAB_simplified_Ikeda.pdf | 344854 | `741f7a85b3170648efd1adda88a73f639ab1590dc033abe050a9bf4d5aed4d65` |
| ikeda/Kawahara_2009_STAB_simplified_Ikeda.txt | 38849 | `01a36bb3e26a3b5bcf670c3a599a6724dcab744a686b32e9c1429cb332ca218d` |
| ikeda/kawahara2009_fig1_markers.csv | 6281 | `462546ad5db9e3341a34bcda347bc9ac10626ba202c316163cb157c1cb806c3e` |

## ittc/

- **What**: ITTC Recommended Procedures: 7.5-02-07-04.5 Estimation of Roll Damping (Rev 01, 2021) and 7.5-02-06-06; .txt files are text extractions.
- **Source**: https://ittc.info/media/4182/75-02-07-045.pdf (roll damping). The download URL of 7.5-02-06-06 was not recorded.
- **Licence**: (c) ITTC; local reference copy, not for redistribution
- **Used by**: hydro/ikeda.py verify() parses Tables 1-5 of the roll-damping procedure

| file | bytes | sha256 |
|---|---:|---|
| ittc/7.5-02-06-06_manoeuvring_benchmark.pdf | 152123 | `34bbb5465e6d8526bb83f326d58ccb53f13059ccbfea828517c781c98ea12694` |
| ittc/7.5-02-06-06_manoeuvring_benchmark.txt | 28522 | `7acdb2cffbfd52de3e0ad63536aa8841fc6f7cfafd266b0b74ac0ae3a95f72c1` |
| ittc/7.5-02-07-04.5_roll_damping.pdf | 591266 | `b66384f5e9742880cd165d9933fe00aa9ce113337fbdb5215cda7922b16d404f` |
| ittc/7.5-02-07-04.5_roll_damping.txt | 138682 | `a96f7b5feeffb7f0a1359cceb3613fbf0f626573b6c9ead0deb16a22b0222fc0` |

## kcs_geometry/

- **What**: KCS hull (and rudder) IGES for T2015 cases 2.10 and 2.11, full scale, millimetres; x forward from the AP, z up from the keel, both sides.
- **Source**: Tokyo 2015 workshop, https://t2015.nmri.go.jp/kcs_gc.html ('Hull & Rudder Geometry (for case 2.10 and 2.11)': KCS_hull_Case2-11.zip)
- **Licence**: no licence stated (workshop benchmark data)
- **Used by**: studies/exp_kcs_t2015_seakeeping.py, studies/exp_hull_robustness.py

| file | bytes | sha256 |
|---|---:|---|
| kcs_geometry/KCS_hull_Case2-11.zip | 558122 | `8cb1e08e4400e307d0f87631f99ea74fca23903a370974d3a458323957a49a3c` |
| kcs_geometry/KCS_hull_Case2-11/FinalHull_KCS.igs | 2809238 | `6e84ad178ea5c2ef62d300bbb8660136d6bb9d8b8fb6112183a9887b8190c416` |
| kcs_geometry/KCS_hull_Case2-11/memo.txt | 33 | `390449859650f7c79e7a8654ce23d2098b5cbf70de0e70872a313e367ddbc823` |

## kcs_t2015/

- **What**: T2015 KCS case 2.10 (head waves) and 2.11 (oblique waves) EFD and CFD time histories over one encounter period (.dat; .lay are Tecplot layouts, unused), and the workshop's condition spreadsheets with the EFD first-harmonic amplitudes and phases (.xlsx). .DS_Store is macOS metadata from the zip.
- **Source**: https://t2015.nmri.go.jp/Instructions_KCS/Case_2.10/ (Case2.10-2.zip, [Identifier]_6conditions_2-10_20150914.xlsx) and .../Case_2.11/ (Case2.11-2_20151112.zip, [Identifier]_6conditions_2-11_20151112.xlsx)
- **Licence**: no licence stated (workshop benchmark data)
- **Used by**: studies/exp_kcs_t2015_seakeeping.py

| file | bytes | sha256 |
|---|---:|---|
| kcs_t2015/[Identifier]_6conditions_2-10_20150914.xlsx | 52160 | `44b0b9e8d2868d8005e02d26e74d24049e9e5d85457c3b577864e2b09f010c68` |
| kcs_t2015/[Identifier]_6conditions_2-11_20151112.xlsx | 28689 | `53cc58279ce3173d12924914b54c3710d175242a61299d83f611505028bb9e4a` |
| kcs_t2015/case2.10/CFD_CT_T-his_C1_2-10.dat | 1920 | `237f210db1f66a5ba104e44f56b1255740565151a517bfc62e2992ec04c3db23` |
| kcs_t2015/case2.10/CFD_CT_T-his_C2_2-10.dat | 1941 | `37a2659c80d7fc868d101f54467d12891a0d7f6b038f27f73ffeb7f213bb6361` |
| kcs_t2015/case2.10/CFD_CT_T-his_C3_2-10.dat | 1920 | `62b083dc4cc37b881d50550e0a043bd171c99470ab8c2265b8348880406acb81` |
| kcs_t2015/case2.10/CFD_CT_T-his_C4_2-10.dat | 1951 | `4d15e3e5e6371892df73a2035c47b8a71fe4ca6a55e12a2db9698537e885f127` |
| kcs_t2015/case2.10/CFD_CT_T-his_C5_2-10.dat | 1963 | `537f72d2c98c49ade1d036428692851bcc84bee2761c15aea51572df1bc65f12` |
| kcs_t2015/case2.10/CFD_Heave_T-his_C1_2-10.dat | 2131 | `66b22798544184c878a838a0d1f14adf22afdd0550f89775219a3e97e0a1cdff` |
| kcs_t2015/case2.10/CFD_Heave_T-his_C2_2-10.dat | 2131 | `4b90dffc277884c24eb5d024e1e748c9564b772c044048fa7b1d89b704387886` |
| kcs_t2015/case2.10/CFD_Heave_T-his_C3_2-10.dat | 2087 | `6a1753b162554b1aa132281b116f64070a4d502a0155adc22f794866952481fd` |
| kcs_t2015/case2.10/CFD_Heave_T-his_C4_2-10.dat | 2085 | `eddae94136a90e362db72b800a49a11ee98bc052353ace8605aaf96c54e71d46` |
| kcs_t2015/case2.10/CFD_Heave_T-his_C5_2-10.dat | 2085 | `e2aa1ae9022447a2803858946c700e050e00d7606ae1705415fbaad840e67dc5` |
| kcs_t2015/case2.10/CFD_Pitch_T-his_C1_2-10.dat | 2030 | `0324c6e42e80bf2c07749f0f888710212e9f698fa53a51fd5a419ba31e093734` |
| kcs_t2015/case2.10/CFD_Pitch_T-his_C2_2-10.dat | 1995 | `40404382f4ed60c8bbbf0b79c3cb97426095ae826aac838b1f7be79a0f6bc5b9` |
| kcs_t2015/case2.10/CFD_Pitch_T-his_C3_2-10.dat | 1981 | `bf5bd476259e4987b365dc257fd31deddb87b19e94457af07e70b996dd1ac7eb` |
| kcs_t2015/case2.10/CFD_Pitch_T-his_C4_2-10.dat | 1980 | `34dc6aba0e7fc76deb6e6a623dc921d8f58c32904afb4d356c0aeaba00342ddb` |
| kcs_t2015/case2.10/CFD_Pitch_T-his_C5_2-10.dat | 1979 | `d42612fcf442cf2c9e685d0ec174a9d447b8daf970603ee33d97e2116458b99c` |
| kcs_t2015/case2.10/EFD_CT_T-his_C1_2-10.dat | 1920 | `b5c2a278955783487bfdbf2ead945e9c96e66751c9dcbaec78933d15c6c18ed1` |
| kcs_t2015/case2.10/EFD_CT_T-his_C2_2-10.dat | 1941 | `efbfeb38e8b769830e24804bc0fc2ce97b64977afe3ac2d91abdcc1e0e9646a9` |
| kcs_t2015/case2.10/EFD_CT_T-his_C3_2-10.dat | 1920 | `917036cb39de0089b248f6145c4240c6e22da642fb52519873f67622b844437d` |
| kcs_t2015/case2.10/EFD_CT_T-his_C4_2-10.dat | 1951 | `73ec96b5f3fd290ca0c1945b8dd81a911e0499acdfcb37da3ecbd4d1d1b8da12` |
| kcs_t2015/case2.10/EFD_CT_T-his_C5_2-10.dat | 1963 | `ce6a0e21bc4b70482760d7a3a826d3aada0bc3d329092b93cded1e6b2695701b` |
| kcs_t2015/case2.10/EFD_Heave_T-his_C1_2-10.dat | 2131 | `605bc4f60349583f23c9da36add30b272f62569b1827893024228e5dec15ef73` |
| kcs_t2015/case2.10/EFD_Heave_T-his_C2_2-10.dat | 2131 | `39a92f54f49e5511cd80f2110faef72187e142fff8fe047e905d3d7844692c9b` |
| kcs_t2015/case2.10/EFD_Heave_T-his_C3_2-10.dat | 2087 | `6d31c77d623712a3028025afa674ce6ab846a91a707e0b8d67d465ebe9e5f62e` |
| kcs_t2015/case2.10/EFD_Heave_T-his_C4_2-10.dat | 2085 | `994f1f7644585d1998e427f1e3da83fdc15bd7c467d80e8d55769842ec8508db` |
| kcs_t2015/case2.10/EFD_Heave_T-his_C5_2-10.dat | 2085 | `acf097bc034fe99b50ae95d1b6205ab71a3b28d4f8971d6ae4a757599f357280` |
| kcs_t2015/case2.10/EFD_Pitch_T-his_C1_2-10.dat | 2030 | `8778c362a03c71cd53ff863d967de564f4706dd52c0dedbfb73676d01229befa` |
| kcs_t2015/case2.10/EFD_Pitch_T-his_C2_2-10.dat | 1995 | `5817839aaee0c7d7348fcff638d26cf478d24b0ca37e2f5d1157396609fb70c6` |
| kcs_t2015/case2.10/EFD_Pitch_T-his_C3_2-10.dat | 1981 | `78cb79fa9744fc3d85a4b2e5e32cbda0cad988888528689bb545486f2b35e69f` |
| kcs_t2015/case2.10/EFD_Pitch_T-his_C4_2-10.dat | 1980 | `60145d7827d091d217ee8a84958e0a3d2512428008350eddacb4585bd17fb50d` |
| kcs_t2015/case2.10/EFD_Pitch_T-his_C5_2-10.dat | 1979 | `7478a8d899c49fc440974352a792674e17e6531902024db2650f66bc8a2167ba` |
| kcs_t2015/case2.11/Case2.11-2_20151112/.DS_Store | 14340 | `7399f0dfc2ec733f9ff034db9212aacf8edb96b4f9605c8be7c614c1579a3f30` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/CFD_CT_T-his_C1_2-11.dat | 2033 | `c973246b0c0fabdb817c937dbc4d33b39a62a2230c627a4717f36532bee2493a` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/CFD_Heave_T-his_C1_2-11.dat | 2103 | `f85497a7edfe6b3dc262b875984e652b6cd95584a10c48018825cf5c851c717e` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/CFD_Pitch_T-his_C1_2-11.dat | 2087 | `977b22ed4fee5e254b56fcf84b6043686d82d1e1b025826a8ff6122059496655` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/CFD_Roll_T-his_C1_2-11.dat | 2144 | `7e3f1e1dea9b2e35f36fea1f07d766d737be95bc006223b0a96f914a8e9def6b` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/CFD_Surge_T-his_C1_2-11.dat | 2035 | `03a9c0e88e40b01a872dded2166b25f1a2572b81873aff5d7f95083a14997bd9` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/EFD_CT_T-his_C1_2-11.dat | 2033 | `fff5c3c8f1ec02750efc568ea21c2ca5dbe53dc8bc37a6ec7c58488e541153e6` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/EFD_Heave_T-his_C1_2-11.dat | 2103 | `5d1d2dde352ff1250ea4dd0eeec4e2d02b6dd782fd0b85e8984ef2cf7a575ba1` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/EFD_Pitch_T-his_C1_2-11.dat | 2087 | `9eed853ab76992985f5fbdb8d767a96f28c909d11d01927c75b2732288134a87` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/EFD_Roll_T-his_C1_2-11.dat | 2144 | `918077a7e1ef3c4fd7f0b9ca07e5f4c3e7d38c901dbd9587bf776581c312c487` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c1/EFD_Surge_T-his_C1_2-11.dat | 2035 | `cee414ecdfa58470bf26345f25ef134d7b3908239cba1aa024475a0c05d5bbba` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/CFD_CT_T-his_C2_2-11.dat | 2033 | `957e986e178be0b7c54f5b256c7897eb17373ecfcdd0246f12dfa6dd70b14d5d` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/CFD_Heave_T-his_C2_2-11.dat | 2095 | `ec11bc8b3f0d320560e6285a13a9a5036d512347e79b1c671c4974b6fe8cf3af` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/CFD_Pitch_T-his_C2_2-11.dat | 2097 | `3f788762658c72ea481564216a6be70806716a80e3db14d0eecf2ee3eb8f3462` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/CFD_Roll_T-his_C2_2-11.dat | 2121 | `b7c75b7a0d191d97dd1866089468736a0d8732cbc30cb7b3de71f0f7f8ba24dd` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/CFD_Surge_T-his_C2_2-11.dat | 2071 | `c998df464bf72e609f811806a30a42e1ed36e73800f116c3998681d5defe4f9a` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/EFD_CT_T-his_C2_2-11.dat | 2033 | `a888a084f6c1f67a385b2f60e7576d57964da7d657285c83ee9127b28baa24fb` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/EFD_Heave_T-his_C2_2-11.dat | 2095 | `c509c8d3ef4021d719e57ad37849bf72ba3a44a333e58fe3dd5ef227f2178ab5` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/EFD_Pitch_T-his_C2_2-11.dat | 2097 | `61dccf43da58f1003ab7ac5da71ac85ac7593fd48cf7a9b8edd9e5601fc6f589` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/EFD_Roll_T-his_C2_2-11.dat | 2121 | `d3f10d4a63e6eab3db2c2c491a27ed814fe692f50d4c1c3d963146d3259dff33` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c2/EFD_Surge_T-his_C2_2-11.dat | 2071 | `d54531cb8c1908ed9431b08ce921c84ebdb88aa724472533e2b34e7a578db877` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/CFD_CT_T-his_C3_2-11.dat | 2033 | `0963496b959fd2f25700a83d2861e09633c4632bc08a86be3916ff5eb54e921d` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/CFD_Heave_T-his_C3_2-11.dat | 2088 | `b6c2ab173940c1370678918de38087a273781397badab1771231a12a0863d4c6` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/CFD_Pitch_T-his_C3_2-11.dat | 2071 | `3ffe8518925483c16d7462e01539bcf7d1f39d824a2e2f71505bbb5548c2583d` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/CFD_Roll_T-his_C3_2-11.dat | 2125 | `639348bb6bd9d1c9ca753a80d3ffc1f5b912c2421929c24e9ae71a77d774ad9c` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/CFD_Surge_T-his_C3_2-11.dat | 2084 | `d7c228fd5657d1f435082109c8d77199f391421c979c3f377849b28cbd8ed63f` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/EFD_CT_T-his_C3_2-11.dat | 2033 | `1bb3569c7054b3ae6ca5fbd1def27f2a4af880a03161dfecff8f7d02f763c95d` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/EFD_Heave_T-his_C3_2-11.dat | 2088 | `8e418c0f2d1b53897583a460b09eb61ff17cef6969d94db724c66bd6f0928101` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/EFD_Pitch_T-his_C3_2-11.dat | 2071 | `d77c70ca1e9e562aff6f2db2cf5b4b0983e3ef65dd964a679cfb1277b9ba33e0` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/EFD_Roll_T-his_C3_2-11.dat | 2125 | `bb6634333c92c395ae5e09cccbefad7f857cabfba45aa752726eb48527dbf35a` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c3/EFD_Surge_T-his_C3_2-11.dat | 2084 | `6f9636832eec5787847037e50d7771925b38eaec3e575167cdd8545c20980092` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/CFD_CT_T-his_C4_2-11.dat | 2033 | `e302d0d4a71b7778934650c482e2bc2fe0663acef67c6d2b99e33680fc2e826e` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/CFD_Heave_T-his_C4_2-11.dat | 2090 | `1534bcfef8c9d621d73bb2f22f3f3ba14277277a57ea68405b5d3f2cdb57c618` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/CFD_Pitch_T-his_C4_2-11.dat | 2089 | `48732dcdcf48555eae586e2465a68057056575eae8e762c816d7055fbc722eac` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/CFD_Roll_T-his_C4_2-11.dat | 2095 | `e56cbf1388a8eed4821bf059b9d2a691ee5abd5c049fc0bbd94a8ec071287410` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/CFD_Surge_T-his_C4_2-11.dat | 2104 | `38c4bd9544fbd1ed1d1ff0f7558522cc5b1d20e245f8093054e50f5a558ea3a5` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/EFD_CT_T-his_C4_2-11.dat | 2033 | `7db70b633552c2e61d1caede83e6270b63b7b2aeebfdd9f5ee3b2fd79b69d9cc` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/EFD_Heave_T-his_C4_2-11.dat | 2090 | `5c42e9c32be30ca862306134b41283612e6aad0ab53dba9b32acca7aeed5e440` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/EFD_Pitch_T-his_C4_2-11.dat | 2089 | `cb30f93664860da0de033d3c1b267bfa93ac67f64229269a82cf45f9e52fdcde` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/EFD_Roll_T-his_C4_2-11.dat | 2095 | `19b50862bd5179088b81ab67bbb702af29b01a6f4f3643eb11a1a029ff2278c7` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c4/EFD_Surge_T-his_C4_2-11.dat | 2104 | `095743858ad6d6a3e6503a334199d9e5cfaae496709a65aed3809d56fe4e6f9e` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/CFD_CT_T-his_C5_2-11.dat | 2033 | `33992bce4121a0d4a88c9ca414136c401b042e253093af56cc0fbfca25f5c954` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/CFD_Heave_T-his_C5_2-11.dat | 2141 | `5d0cb7a4f3742af975ee7f860e5c30496762138d62575c6229a0251e65e03c76` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/CFD_Pitch_T-his_C5_2-11.dat | 2083 | `57109e7fa1613cf9450b5497abcbd42e273b80480b83967d008b55053a39ea55` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/CFD_Roll_T-his_C5_2-11.dat | 2114 | `7a742bca2bddaaa19cb6e5728b2dca642ec6253a9781925dce7d1b3e6c857a50` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/CFD_Surge_T-his_C5_2-11.dat | 2095 | `92f8d0ae72fa92ca960d54f65e3fe23cc5f06c326a1392e788678e35e5676774` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/EFD_CT_T-his_C5_2-11.dat | 2033 | `e51cdd0c97eb25674e1977c0364c03b1fa1a601fc1368fb39d2278db3f48a0cc` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/EFD_Heave_T-his_C5_2-11.dat | 2141 | `05199b61f24f3ba92aec88b82458ca04ef06efd43eb57452394969cc55526bee` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/EFD_Pitch_T-his_C5_2-11.dat | 2083 | `240246f984d8150eec36f51e70e7fdb00e3c9d42d1fc51172712006f78544352` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/EFD_Roll_T-his_C5_2-11.dat | 2114 | `75ad2abf7c42def3689ce1fa40ad2293d2287c43cc20695524fcf985e602ae7f` |
| kcs_t2015/case2.11/Case2.11-2_20151112/c5/EFD_Surge_T-his_C5_2-11.dat | 2095 | `520886569ff11a64d4d31d8f8628a292446c7a14cce3ad1abdeca05a72370790` |

(25 Tecplot .lay layout files not listed; unused.)

## kvlcc2_geometry/

- **What**: KVLCC2 hull IGES, full scale, metres, HALF hull; x AFT from the AP, z DOWN from the keel (established by the LCB sign and the flat-bottom position, hydro/cad_import.py); the flat bottom is in the file twice.
- **Source**: https://simman2014.dk/wp-content/uploads/2015/07/KVLCC2-Hull.zip
- **Licence**: no licence stated (SIMMAN benchmark data)
- **Used by**: hydro/cad_import.py main(), studies/exp_kvlcc2_roll_prediction.py

| file | bytes | sha256 |
|---|---:|---|
| kvlcc2_geometry/KVLCC2-Hull.zip | 1345336 | `dc032647f00a68d354ede2f4b1ee3d68cd048bc3d34d221c2a8c9aac3a6dd548` |
| kvlcc2_geometry/kvlcc2.igs | 6270129 | `ed06c287c7a858a182494cb101e2c322bb2c407994c646a95d9e9f811638e2f9` |

## kvlcc2_propeller/

- **What**: KVLCC2 propeller open-water data (NMRI model; HMRI sheet).
- **Source**: https://simman2014.dk/wp-content/uploads/2015/07/KVLCC2-Propeller-Openwater-Data-NMRI-Model.zip and ...-HMRI.pdf
- **Licence**: no licence stated (SIMMAN benchmark data)
- **Used by**: studies/exp_propeller.py

| file | bytes | sha256 |
|---|---:|---|
| kvlcc2_propeller/KVLCC2-Propeller-Openwater-Data-HMRI.pdf | 6432 | `373304df16cf910f95c28e0ce2c6be163cd44e1ae219798dd1c25da4e06a9f4a` |
| kvlcc2_propeller/KVLCC2-Propeller-Openwater-Data-NMRI-Model.zip | 672 | `c04722b18cfcd3c4a4600084147ef8f636aea5098fe398e58999a2e5e54adfed` |
| kvlcc2_propeller/KVLCC_POT_NMRI.txt | 891 | `5c0d43113ecd03af3fbcdfbcd92cd31502c9437ad84e5a79816b90d5e47c1cb3` |

## kvlcc2_rolldecay/

- **What**: SSPA roll-decay model tests of KVLCC2 at 1:68 (0 kn x2, 15.5 kn), with loading condition and FNPF parameters (Alexandersson & Kjellberg).
- **Source**: https://data.mendeley.com/datasets/2stvkyngj9/1
- **Licence**: CC BY 4.0
- **Used by**: studies/exp_kvlcc2_rolldecay.py, studies/exp_kvlcc2_roll_prediction.py

| file | bytes | sha256 |
|---|---:|---|
| kvlcc2_rolldecay/description.txt | 620 | `5af289a3561f57d7d88350088a3c58248b906ed4cede6b9e70052f8cad8c1793` |
| kvlcc2_rolldecay/fnpf_parameters.csv | 3061 | `637207095f207e171f79a466f02f521c986263ae22515991786f7d6d4df90b0c` |
| kvlcc2_rolldecay/model_test_21337.csv | 1279618 | `9a5ed4a7a40b988fafc35f3ad984a3332dd819b4c7b7718c17a37aee56d3dd0c` |
| kvlcc2_rolldecay/model_test_21338.csv | 2791060 | `015dff27cb20eb84b31aa4565652def77bfb7161c91b1250a481a41c2a666018` |
| kvlcc2_rolldecay/model_test_21340.csv | 1259770 | `09308797d437e2515c5d1ee749819f489cd8636a01bd2434a97209583bdcb30f` |
| kvlcc2_rolldecay/model_test_parameters.csv | 1801 | `1d792a890d968dd6a5df3ebf2cc7a0889e2e350a52282f8a8ee8109140a6f617` |
| kvlcc2_rolldecay/model_test_units.csv | 48 | `83b8d4ad3438dc15784a1c34ba8e610ac8379646bf40f4c61ed977ef59e01612` |

## wigley/

- **What**: Wigley parabolic hull (L/B 10, B/T 1.6) measurements of the Japanese cooperative experiment (Kajitani et al. 1983, "The summary of the cooperative experiment on Wigley parabolic model in Japan", Proc. 2nd DTNSRDC Workshop on Ship Wave-Resistance Computations; DTIC ADP003037, not retrievable here: HTTP 403), in two secondary forms, for Track B step B4 (validation rows V1-V5, `studies/rankine_hull_verify.py`; stage note `learn/meta/RANKINE_STAGE1_2026-10-06.md` section 9).
  1. `wavebem_*`: files shipped with WaveBEM (github.com/mathLab/WaveBEM, `experimental-results/wigley/`, local clone `ext_sims/WaveBEM`, commit b44b0fb 2017-03-07; LGPL-2.1 repository). `wigleyDragExp1-4.csv`: C_W of the model held FIXED vs Fn, labelled SRI, UT1, UT2, UT3 in the files (series 4 is a third U. Tokyo series; Y.N.U. -- "Y.M.U." in the legend -- is NOT digitised [corrected in the B4 review; the first version said series 4 = Y.N.U.]), digitised by the WaveBEM authors from Hendrix & Noblesse's (1988) figure (copy shipped as `tokyoWigley.pdf`, page "- 61 -" upper panel, "Wave-resistance coefficients C_W & C_WP for the Wigley hull held in fixed position"; their data reference: McCarthy 1985, DTNSRDC-85/011, which collects the Kajitani et al. 1983 data). **Transcription defect (B4 review, stage note section 9.7.1, overlay `studies/_cache/rankine/b4_cw_overlay.png`):** at Fn < 0.3 the WaveBEM points follow the zeroth- and first-order slender-ship theory lines of that figure, on average +0.16 ... +0.28e-3 above the U. Tokyo vector curve of Yang et al. (2000), which passes through the U. Tokyo markers of the same scan; these four files are not a valid C_W reference there. `tokyoFr*.csv`: wave profiles 2 g zeta/U^2 along the hull (fixed model) at Fn 0.25, 0.267, 0.289, 0.316, 0.354, 0.408 (Curve1 = U. Tokyo; Curve2 at Fn 0.25 = S.R.I.), digitised from page 62 of the same PDF; x in WaveBEM's frame (bow at x/L = -0.5: our x = -x_file). `wigleyDrag.csv`: the four series resampled on a common Fn grid (not used). Digitisation of scanned figures: error not stated by the source.
  2. `yang2000_*`: curves of Yang C., Lohner R., Noblesse F., Huang T.T. (2000), "Calculation of ship sinkage and trim using unstructured grids", ECCOMAS 2000, Barcelona, paper 367 (congress2.cimne.com/eccomas/proceedings/eccomas2000/pdf/367.pdf, fetched 2026-10-06, copy `yang_lohner_noblesse_huang_2000_eccomas367.pdf`): Figure 4a (sinkage s and trim t of the model FREE to sink and trim) and Figure 5a (wave drag coefficient, fixed and free), "Experiment Results" = U. Tokyo (paper section 5). The figures are vector graphics: the curves were read from the PDF drawing commands by `studies/rankine_wigley_data.py` (calibration by the tick labels snapped to the tick lines; label residuals <= 2e-5 of the axis values), so the transcription error is the PDF coordinate precision, not a digitisation. Definitions printed in the paper (p. 9): s = (dF' + dA')/Lpp (dF', dA' draught changes, > 0 deeper), t = (dA - dF)/Lpp (> 0 bow up). The printed s is twice the mean sinkage; which definition the plotted experimental curve follows is resolved by the registered rule of check V3 (stage note 9.1), not by the source.
  3. `hn1988_p61_*` (added in the B4 review): markers re-read from the scan of Hendrix & Noblesse (1988) p. "- 61 -" (`wavebem_tokyoWigley.pdf`, embedded 300-dpi bilevel image) by `studies/rankine_wigley_scan.py`: per-panel affine calibration fitted to the tick marks detected on the frame lines (C_W panel 2483 px per unit Fn and 118.9 px per 1e-3, residuals 0.0008 in Fn and 0.015e-3; C_WP panel 2479 and 110.2, residuals 0.0005 and 0.022e-3), marker centres refined from approximate positions by the centroid of the opened dark blob; reading error +-3 px for isolated markers (+-0.0012 in Fn, +-0.025e-3 in C_W, +-0.027e-3 in C_WP), +-5 px for markers merged with another marker or a curve. `hn1988_p61_cwp_markers.csv`: all S.R.I. and U. Tokyo wave-pattern resistance (C_WP) markers of the lower panel (13; "SRI+UT" = overlapping markers read as one); `hn1988_p61_cw_markers_fn020.csv`: the C_W markers of the upper panel in Fn 0.185-0.215 only (11; laboratory scatter at low Fn; symbol identification uncertain where marked "?"). Overlay of the WaveBEM series and the Yang curve on the scan: `studies/_cache/rankine/b4_cw_overlay.png`; numbers: `studies/_cache/rankine/b4_scan_reading.json`.
- **Licence**: WaveBEM files LGPL-2.1 (repository licence; unchanged copies, notice in `data/external/wigley/README.md`); Yang et al. 2000: copyrighted conference paper, no licence stated, used for comparison only; the local PDF copy must stay out of any published copy of the repository (`data/external/wigley/.gitignore`); the `yang2000_*` and `hn1988_p61_*` CSV files are our transcriptions (numbers only).
- **Used by**: studies/rankine_hull_verify.py (V1-V5, V1a, V1b, V5a), studies/rankine_wigley_data.py, studies/rankine_wigley_scan.py

| file | bytes | sha256 |
|---|---:|---|
| wigley/hn1988_p61_cw_markers_fn020.csv | 741 | `0be1e67eb698737adefae704ee6c19dc13e8d868dfe77ed225741b61c4fba7b7` |
| wigley/hn1988_p61_cwp_markers.csv | 688 | `a483941c4bfe65eb014e1c519355e63e9fe77bd40a975e06a73f341e375e029a` |
| wigley/README.md | 2180 | `81d65bee8f06ddb5442200d095e2f398f056384d3c89312c8458debbfe134ea1` |
| wigley/wavebem_tokyoFr025.csv | 1474 | `b2c0536680351b7071a085ce5d0b2c77f00b6a9dc2143f47126faf2e799752a3` |
| wigley/wavebem_tokyoFr0267.csv | 1426 | `3cac344b578fb25ed5144e714bbc8c8900c5f0fd7a16c5cd2ca5b9c13adc3005` |
| wigley/wavebem_tokyoFr0289.csv | 1420 | `dc5661884d074f8f4ae4dd7b3452877fa8110ab5d36276aa68585619c0a6c59d` |
| wigley/wavebem_tokyoFr0316.csv | 1423 | `d895d0bd53f0b4f919e6d1dd1f1e9c04bab2f3aefe9aea0ad9c2e6653fc5aa61` |
| wigley/wavebem_tokyoFr0354.csv | 553 | `3b0a9a41cbc50dad2a53a112c0c77b3f368e08fb332048b2717a2af189ff93dc` |
| wigley/wavebem_tokyoFr0408.csv | 551 | `680a126e2ac7222587b3b4f414f8c09282ad67a4e03d8373a2652622c9ca3af5` |
| wigley/wavebem_tokyoWigley.pdf | 177321 | `d64e9eb9f1a1167b679218f959d7904751c0430d003e843f20db712a1f0abd27` |
| wigley/wavebem_wigleyDrag.csv | 3648 | `2f190becc4728171600b74ac7eec300b29ee046b877aa1eb433552159b45dad6` |
| wigley/wavebem_wigleyDragExp1.csv | 457 | `0ec72f5a81f2c71d5546faac0370b9125bfeddf5b7017138a0afde9724b76a8e` |
| wigley/wavebem_wigleyDragExp2.csv | 285 | `ae134ac72174f21ccbd46d7cbd0ab2240741a6bdac6f3db7549abddbf4237237` |
| wigley/wavebem_wigleyDragExp3.csv | 327 | `6a3a1fc3bd439549db755ce480490649b1bebc34a34fbcfc8419f6a075cf82b5` |
| wigley/wavebem_wigleyDragExp4.csv | 351 | `263a2ef9c3d6177df4a5adb7b2b400607b963c4ec84efd3d08221a80073f9fea` |
| wigley/yang2000_fig4a_sinkage_s.csv | 562 | `8b96811c017d830dd5799dd447e625e8d4e7dd696ad8771924951483965e4533` |
| wigley/yang2000_fig4a_trim_t.csv | 571 | `06e107db69cc1f91349292e8f54e229cc78763d5287d91aa450e87a7fcbac22b` |
| wigley/yang2000_fig5a_cw_fixed.csv | 565 | `339425fd52fa59964caeecfe8f80f3c56a20982e6c2958ff28d9c7b216878769` |
| wigley/yang2000_fig5a_cw_free.csv | 578 | `8f361ec4d95561f72f3d1e617a27ce5ba700304c83e74f030650e49817f9c7c6` |
| wigley/yang_lohner_noblesse_huang_2000_eccomas367.pdf | 358967 | `381a77558581eadaf6ba8fb6b46c3344b22d3c3cc17e4e3f0cb7a350398cd7ee` |

Not obtained (V3): the original Kajitani et al. 1983 tables (DTIC ADP003037, blocked), Journee 1992 (TU Delft report 909; repository returned HTTP 403), Doctors, Macfarlane & Young 2007 (International Shipbuilding Progress 54:145-163, paywalled) for the transom-ventilation regression. None was indispensable for B4: the first two are replaced by the secondary sources above (with the sinkage-definition rule), the third is registered as quoted in WaveBEM with an assumed scatter (EMPIRICAL_REGISTER_RANKINE.md, B4-1).

## wigley_journee/

- **What**: forward-speed seakeeping experiments on the modified Wigley hull "Wigley I" (2 y/B = (1 - zeta^2)(1 - xi^2)(1 + 0.2 xi^2) + zeta^2 (1 - zeta^8)(1 - xi^2)^4, L 3 m, B 0.3 m, T 0.1875 m, V 0.0946 m^3, k_yy 0.25 L) of J.M.J. Journee (1992), "Experiments and calculations on four Wigley hullforms", TU Delft, Ship Hydromechanics Laboratory, report 909 -- **original not obtained** (repository.tudelft.nl, record uuid:52a76d0b-5db1-4902-a254-a81491b9cf99, HTTP 403 to scripted access, file listed under an indefinite embargo). Used by Track B step B5 (validation rows V1-V4, `studies/rankine_unsteady_verify.py`; stage note `learn/meta/RANKINE_STAGE1_2026-10-06.md` section 10) as SECONDARY transcriptions: the experimental markers that two open-access papers plot for Journee's values, read from the PDF drawing commands of their vector figures by `studies/rankine_journee_data.py` (per-panel affine calibration from the axis tick marks or tick-label glyph centres; marker position = bounding-box centre; reading uncertainty = 1/6 of the marker size + calibration residual, written per point):
  1. `journee1992_wigley1_fn020_*.csv` (Fn 0.2, head seas): J. Mar. Sci. Eng. 2021, 9, 87, "Hydrodynamic analysis and motions of ship with forward speed via a three-dimensional time-domain panel method" (MDPI, CC BY 4.0; PDF from mdpi-res.com fetched 2026-10-06, copy `jmse-09-00087.pdf`), PDF pages 17-19: Fig. 12 (A33' = A33/(rho V), B33' = B33/(rho V) sqrt(L/g)), Fig. 13 (A55' = A55/(rho V L^2), B55'), markers "Experiment" (blue down-triangles) = their ref. 29 (Journee 1992); Fig. 14 (heave and pitch exciting-force amplitudes F3' = F3 L/(rho g eta0 V), F5' = F5/(rho g eta0 V) vs nu' = k L; normalisation checked by the hydrostatic limit A_w L/V = 19.8 at k -> 0); Fig. 16 (RAOs zeta3/eta0 and zeta5 L/(2 pi eta0) vs lambda/L). The plotted abscissae of figs 12-13 deviate from Journee's test frequencies (omega = 3 ... 10 rad/s, omega' = 1.659 ... 5.530) by up to 0.07; the point A33'(1.68) = 1.63 is about twice its neighbours (probable transcription outlier, flagged in the registration).
  2. `journee1992_wigley1_fn030_*.csv` (Fn 0.3): F. Kara (2022), "Time domain potential and source methods and their application to twin-hull high-speed crafts", Ships and Offshore Structures (open-access version of record, Sheffield Hallam repository, shura.shu.ac.uk/29602, fetched 2026-10-06, copy `kara2022_shura29602.pdf`), page 8, Fig. 9 (A33/(rho V), B33/(rho V sqrt(g/L)) vs omega sqrt(L/g)), markers "Experiment - Journee (1992)"; abscissae equal Journee's test frequencies to 0.01.
  - Not used (raster figures only): arXiv 1401.3816 (Weymouth & Yue), J. Ocean Mech. Aerosp. Sci. Eng. 22 (2015) (Excel charts, inconsistent hull data), J. Mar. Sci. Eng. 8:106 (raster).
- **Licence**: CSV files are our transcriptions (numbers only); source PDFs kept for traceability only and excluded from any published copy (`data/external/wigley_journee/.gitignore`; the MDPI article is CC BY 4.0, the licence of the second copy was not verified).
- **Used by**: studies/rankine_unsteady_verify.py (V1-V4), studies/rankine_journee_data.py

| file | bytes | sha256 |
|---|---:|---|
| wigley_journee/README.md | 1584 | `1a326ae99c8b06fed7c09376a175307149c8d31c201a75bc313afeedca6b8e6e` |
| wigley_journee/.gitignore | 119 | `78e8e0fe93b45967d91b04ffd027de2bf9e8043dd7bd599ba36ce6212f70dbf2` |
| wigley_journee/jmse-09-00087.pdf | 3315714 | `13b6af444b9b57609eb83854c3c5f3a842863617e18f5599be34f171a337fdbc` |
| wigley_journee/kara2022_shura29602.pdf | 5036981 | `363230e31d9752a62dad5fbfe4823718ac2af0e81c7d549d2ad53e8aa5a12c80` |
| wigley_journee/journee1992_wigley1_fn020_A33.csv | 568 | `e2567c85beb7eea71baac2f84760ed3ba8395283073b0e666dbf162bf7abf260` |
| wigley_journee/journee1992_wigley1_fn020_A55.csv | 568 | `8d76356fab70e7be0de2dea6be1d5b8e533478b4bd983ed45eafe6e616c56e29` |
| wigley_journee/journee1992_wigley1_fn020_B33.csv | 568 | `b2d7029009545b8efa21d98075c18508f33d6926b65db46f363f947e831ad2f4` |
| wigley_journee/journee1992_wigley1_fn020_B55.csv | 535 | `daf078e9422c3aa72cb9ba6d1d7efea6fa0675a5ba04354e3418bc4c16b58dc2` |
| wigley_journee/journee1992_wigley1_fn020_F3.csv | 528 | `7c313dbc4c15d981348379ab01db176806846181139e348d8dfb52f2395db708` |
| wigley_journee/journee1992_wigley1_fn020_F5.csv | 527 | `b3ce9a85257d9bf0e0306b51f98bfcf8c7d8163eccfc26937353a71ac857a65e` |
| wigley_journee/journee1992_wigley1_fn020_RAO3.csv | 513 | `01638cae9555b441f7ae5659c383010f32b1ad55a4823c38bd7bf7d78c01621a` |
| wigley_journee/journee1992_wigley1_fn020_RAO5.csv | 513 | `d03c38193df188a1a70e9f83fec907306896e7eca72d4660f73589c7a3dd8987` |
| wigley_journee/journee1992_wigley1_fn030_A33.csv | 563 | `859041b8330cf2c0384c057051406496d093ea1df3eae9f424f2684fd48115e1` |
| wigley_journee/journee1992_wigley1_fn030_B33.csv | 563 | `878dca221f2331f0b1d262e0501b849ef1bb5e00927774716b212a2c719c04ea` |

## wu_eatock_taylor_sphere/

- **What**: analytical (exact forward-speed Green function, linear theory) added mass, damping and exciting forces of a
  submerged sphere at forward speed from G. X. Wu and R. Eatock Taylor (1988), "Radiation and diffraction of water waves
  by a submerged sphere at forward speed", Proc. R. Soc. Lond. A 417:433-461 -- **original not obtained** (publisher
  page behind a bot challenge, not bypassed; not bought). Used as SECONDARY transcription from the "Analytical results
  [27]" columns of T. Mei, M. Candries, E. Lataire, Z. Zou (2020), J. Mar. Sci. Eng. 8(2):106 (MDPI, CC BY 4.0; PDF from
  mdpi-res.com fetched 2026-10-07, copy `jmse-08-00106.pdf`), tables 3, 5, 6 (PDF pages 13-15: A11, B11, A33, B33, A13,
  A31, B13, B31 at k_e r = 0.6-5.0) and tables 1-2 (page 11: exciting forces f1, f3 at kr = 0.4-2.0, with the tables'
  own NK columns as a numerical cross-reference). Sphere radius r, centre depth 2 r, Fr = U/sqrt(g r) = 0.4, deep water;
  normalisation A = a/(pi rho r^3), B = b/(pi rho omega_e r^3). Numbers copied from the PDF text layer (no digitising).
  Used by Track B step B5 review, check WE (`studies/rankine_unsteady_verify.py`; stage note section 10.8.4).
- **Licence**: CSV files are our transcriptions (numbers only); the source PDF is CC BY 4.0, kept for traceability and
  excluded from any published copy (`data/external/wu_eatock_taylor_sphere/.gitignore`).
- **Used by**: studies/rankine_unsteady_verify.py (WE, WEnk, WEx)

| file | bytes | sha256 |
|---|---:|---|
| wu_eatock_taylor_sphere/README.md | 1693 | `0ca7a9b1cc8518f05fa605f48fdd387d9dafbf13b57c36369926e584b4522d1f` |
| wu_eatock_taylor_sphere/.gitignore | 18 | `6bbec4e3327d073c525f19d13acb711c28558442c9d58b39d87900274ebdfc47` |
| wu_eatock_taylor_sphere/jmse-08-00106.pdf | 1536193 | `393621782631c3440b9a9387376abcede133d8d68bf1ce15f22ecec9a3d9f1fc` |
| wu_eatock_taylor_sphere/we1988_sphere_fr040_h2r_radiation.csv | 1700 | `c084c4c825423be211f81dcbed321beec61d8aee6e37ab13da53b44f7118f8c2` |
| wu_eatock_taylor_sphere/we1988_sphere_fr040_h2r_excitation.csv | 1622 | `524c4daec413f9463fd13b0a5cec7f603a8058cf18e41deb1001dbb783904923` |

## (no folder) Savitsky (1964) planing equations -- Track B stage B7 gate

- **What**: the empirical prismatic-planing equations of D. Savitsky (1964), "Hydrodynamic design of planing hulls",
  Marine Technology 1(1):71-95, used as the validation REFERENCE of the planing gate (stage note
  `learn/meta/RANKINE_STAGE1_2026-10-06.md` section 13): lift C_L0 = tau^1.1 (0.0120 lam^0.5 + 0.0055 lam^2.5 / C_V^2)
  (tau in deg), C_L,beta = C_L0 - 0.0065 beta C_L0^0.6 (beta in deg), centre of pressure l_p/(lam b) = 0.75 - 1/(5.21
  C_V^2/lam^2 + 2.39), keel-chine wetted-length difference (L_K - L_C)/b = tan(beta)/(pi tan(tau)); validity tau 2-15
  deg, lam <= 4, C_V 0.60-13, beta <= 30 deg. These are the standard published forms (identical in Savitsky 1964 and in
  Faltinsen 2005, Hydrodynamics of High-Speed Marine Vehicles, ch. 9); they are coded in
  `hydro/rankine/planing_steady.py: savitsky()`, no data file. **Original paper not re-read in this stage** (the
  equations are textbook; the coefficients were cross-checked on 2026-10-07 against published restatements, e.g. J.
  Naval Arch. Mar. Eng. Dec. 2015 (banglajol JNAME 22351) and Int. J. Physics 7(1) (sciepub ijp-7-1-4)).
- **Raw data not transcribed**: Chambliss & Boyd (1953), NACA TN 2876 (prismatic surfaces, deadrise 20 and 40 deg; UNT
  Digital Library item metadc56859, tables I-II on pages 10-14) and Shuford (1958), NACA Report 1355, are available as
  scans whose OCR text layer was checked on 2026-10-07 and is too corrupted (columns merged, digits lost) to transcribe
  reliably without page images; no page image was downloaded. Recorded as a limitation of the gate (stage note 13.6).
- **Used by**: studies/rankine_planing_gate.py (rows VL, VLR, VCP, VW, VEQ, VF).

## wedge_entry -- 2-D wedge water entry (similarity solution, MLM peak pressures, Wen et al. 2022 transition stage) -- Track B stage B8

- **What**: reference numbers for the nonlinear cross-section of the planing route-1 gate (stage note
  `learn/meta/RANKINE_STAGE1_2026-10-06.md` section 14): (1) Zhao & Faltinsen (1993) J. Fluid Mech. 246:593-612,
  constant-velocity wedge similarity solution, peak pressure and its height -- **original not obtained**, secondary
  transcription from Iafrati & Battistin, arXiv 1212.6699 v1, Table 1 (PDF p. 17, column "ZF"); (2) peak pressures of
  the Logvinovich-type models and the exact integral solution, "On wedge-slamming pressures" (J. Fluid Mech.; arXiv
  2011.10378v3 HTML, Table 1, section 2.4.2); (3) constants of the transition-stage force similitude of Wen, Liu, Del
  Buono, Qu & Iafrati (2022) J. Fluids Struct. (arXiv 2201.00959 v1, eqs 16-24, pp. 22-31). All read 2026-10-07 via
  WebFetch / text layer of the arXiv PDFs; numbers typed from the text layer; no PDF kept in the repository.
- **Licence**: our transcriptions (numbers only).
- **Used by**: hydro/rankine/planing_nl.py (E-B8-1 constants), studies/rankine_planing_b8.py (W2A, W2C, W2P, W2Z).

| file | bytes | sha256 |
|---|---:|---|
| wedge_entry/README.md | 889 | `9771ddbaa81482891cc0665c65ee79c0412b780aa80138e97a8a3d82fdaa74db` |
| wedge_entry/zf1993_similarity_wedge.csv | 713 | `0cb67d5410a13b4717806e615756610cd4fb01eee111793c19baa553f45a99bb` |
| wedge_entry/wedge_pmax_models_2011.10378.csv | 654 | `4f5b44ed123f8a0b006f3afe600c49fda6b97c8062191f6f92e9575dc9c965da` |
| wedge_entry/wen2022_transition_constants.csv | 1168 | `78ba06cfded2aeeb2ba5d9fb9127a591aad30315a08530af1474386009ae3690` |

## tveitnes_2008/
- Tveitnes, Fairlie-Clarke, Varyani (2008), "An experimental investigation into the constant velocity water entry of wedge-shaped sections", Ocean Engineering 35. Open-access author version from the University of Strathclyde repository: https://pure.strath.ac.uk/ws/portalfiles/portal/374755/OE-KSV_an_experimental_invest.pdf (fetched 2026-10-09, 2.9 MB, 34 pages). Used for check V5 (post-chine force at cross-plane Froude number 0.3-0.7). Digitised values carry reading error; see studies/_cache/derive/v5_*.

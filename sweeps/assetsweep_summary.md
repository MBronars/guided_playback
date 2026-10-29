# Asset sweep: pick-to-location lift, pickup-only ckpt (cont200k final.pt)

Recipe: 4 members, 39 transitions, S=10, target = f0 centroid + (0.15, 0, 0.15) m, Adam lr 0.1, 10 it, shell ON, seed 0, eager bf16; Isaac floating-hand replay of the guided rollout.
Contact = first frame with min finger(roles==1)-object distance < 5 mm; tilt = Kabsch rotation angle of object points vs frame 0 at the last pre-contact frame (max over pre-contact frames in brackets); disp = pre-contact centroid displacement at that frame. Held = Isaac final rise > 50 mm.
Sorted by #held with gen AND Isaac tilt < 5 deg, then #held with Isaac tilt < 5 deg, then #held.

## Per episode

| episode | asset (id) | extent mm | rec contact f / tilt deg | held | held & Isaac tilt<5 | held & gen tilt<5 | held & both<5 | gen tilt deg (4 members) | Isaac tilt deg (4 members) |
|---|---|---|---|---|---|---|---|---|---|
| peanut_butter_jar_asset_0003 | Kitchen_Table/peanut_butter_jar/asset_0003 (7070) | 57x60x93 | 14 / 0.0 | 3/4 | 3 | 2 | 2 | 0.3, 0.7, 12.6, 6.6 | 0.0, 0.0, 2.3, 0.0 |
| jam_jar_asset_0085 | Kitchen_Table/jam_jar/asset_0085 (6097) | 59x60x105 | 14 / 1.8 | 2/4 | 2 | 2 | 2 | 2.1, 1.3, 15.9, 0.5 | 0.0, 0.0, 5.2, 1.9 |
| batch000_e1 | libero/alphabet_soup_can/asset_0034 [libero id: sorted-order mapping not verified for libero; mesh fit agrees] (9050) | 59x56x73 | 15 / 0.0 | 4/4 | 4 | 1 | 1 | 11.4, 13.0, 13.0, 1.9 | 0.0, 0.0, 0.0, 0.0 |
| mayonnaise_jar_asset_0008 | Kitchen_Table/mayonnaise_jar/asset_0008 (6430) | 121x117x67 | 14 / 4.0 | 4/4 | 4 | 1 | 1 | 1.0, 6.1, 50.1, 49.7 | 3.9, 3.9, 1.4, 3.9 |
| spice_jar_asset_0025 | Kitchen_Table/spice_jar/asset_0025 (8022) | 48x48x95 | 19 / 1.3 | 4/4 | 4 | 0 | 0 | 16.1, 17.9, 23.4, 14.2 | 0.1, 0.1, 0.1, 0.1 |
| cracker_box_asset_0039 | Kitchen_Table/cracker_box/asset_0039 (5289) | 39x107x150 | 28 / 6.9 | 1/4 | 0 | 0 | 0 | 4.8, 13.3, 24.8, 17.1 | 0.3, 0.0, 8.5, 14.3 |
| cereal_box_asset_0044 | Kitchen_Table/cereal_box/asset_0044 (5137) | 59x106x150 | 25 / 0.2 | 1/4 | 0 | 0 | 0 | 23.1, 7.6, 25.5, 18.7 | 85.2, 7.7, 85.3, 85.3 |
| tea_box_asset_0019 | Kitchen_Table/tea_box/asset_0019 (8331) | 98x97x122 | 14 / 0.0 | 0/4 | 0 | 0 | 0 | 5.4, 0.1, 0.9, 0.6 | 0.0, 18.4, 0.0, 0.0 |
| pickle_jar_asset_0025 | Kitchen_Table/pickle_jar/asset_0025 (7257) | 71x70x123 | 16 / 0.0 | 0/4 | 0 | 0 | 0 | 0.5, 0.6, 0.7, 4.4 | 1.6, 1.6, 1.6, 1.6 |
| pasta_box_asset_0058 | Kitchen_Table/pasta_box/asset_0058 (7025) | 128x147x86 | 24 / 13.5 | -/4 | - | - | - | 24.2, 27.6, 15.4, 17.1 | - |
| juice_carton_asset_0025 | Kitchen_Table/juice_carton/asset_0025 (6137) | | | ERROR: no rollout.npz | | | | | |
| milk_carton_asset_0037 | Kitchen_Table/milk_carton/asset_0037 (6574) | | | ERROR: no rollout.npz | | | | | |
| food_can_asset_0039 | Kitchen_Table/food_can/asset_0039 (5711) | | | ERROR: no rollout.npz | | | | | |
| soda_can_asset_0012 | Kitchen_Table/soda_can/asset_0012 (7859) | | | ERROR: no rollout.npz | | | | | |
| tuna_can_asset_0023 | Kitchen_Table/tuna_can/asset_0023 (8519) | | | ERROR: no rollout.npz | | | | | |
| mustard_bottle_asset_0059 | Kitchen_Table/mustard_bottle/asset_0059 (6796) | | | ERROR: no rollout.npz | | | | | |
| water_bottle_asset_0087 | Kitchen_Table/water_bottle/asset_0087 (8788) | | | ERROR: no rollout.npz | | | | | |
| drinking_glass_asset_0056 | Kitchen_Table/drinking_glass/asset_0056 (5563) | | | ERROR: no rollout.npz | | | | | |
| lemon_asset_0012 | Kitchen_Table/lemon/asset_0012 (6404) | | | ERROR: no rollout.npz | | | | | |
| tomato_asset_0012 | Kitchen_Table/tomato/asset_0012 (8478) | | | ERROR: no rollout.npz | | | | | |

## Per member

| episode | m | gen contact f | gen tilt@c (max) deg | gen disp@c mm | gen peak/final rise mm | gen goal mm | Isaac contact f | Isaac tilt@c (max) deg | Isaac disp@c mm | Isaac peak/final rise mm | Isaac goal mm | held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| peanut_butter_jar_asset_0003 | 0 | 12 | 0.3 (0.3) | 0.2 | 139 / 139 | 23 | 13 | 0.0 (0.0) | 0.0 | 115 / 115 | 44 | Y |
| peanut_butter_jar_asset_0003 | 1 | 9 | 0.7 (0.7) | 0.6 | 167 / 167 | 19 | 11 | 0.0 (0.0) | 0.0 | 132 / 132 | 29 | Y |
| peanut_butter_jar_asset_0003 | 2 | 15 | 12.6 (12.6) | 14.4 | 148 / 148 | 11 | 14 | 2.3 (2.3) | 2.3 | 116 / 116 | 39 | Y |
| peanut_butter_jar_asset_0003 | 3 | 13 | 6.6 (6.6) | 4.5 | 146 / 140 | 20 | 13 | 0.0 (0.0) | 0.0 | 93 / -26 | 184 | n |
| jam_jar_asset_0085 | 0 | 14 | 2.1 (2.2) | 1.8 | 131 / 131 | 19 | 14 | 0.0 (0.0) | 0.0 | 156 / 156 | 58 | Y |
| jam_jar_asset_0085 | 1 | 9 | 1.3 (1.3) | 1.4 | 156 / 156 | 16 | 9 | 0.0 (0.0) | 0.0 | 152 / 152 | 36 | Y |
| jam_jar_asset_0085 | 2 | 13 | 15.9 (15.9) | 15.4 | 136 / 128 | 51 | 12 | 5.2 (5.2) | 5.0 | 4 / -26 | 231 | n |
| jam_jar_asset_0085 | 3 | 10 | 0.5 (0.5) | 0.6 | 170 / 170 | 23 | 11 | 1.9 (1.9) | 1.8 | 4 / -25 | 264 | n |
| batch000_e1 | 0 | 19 | 11.4 (11.4) | 11.8 | 120 / 120 | 44 | 18 | 0.0 (0.0) | 0.0 | 127 / 127 | 35 | Y |
| batch000_e1 | 1 | 15 | 13.0 (13.0) | 12.3 | 155 / 155 | 28 | 14 | 0.0 (0.0) | 0.0 | 142 / 142 | 25 | Y |
| batch000_e1 | 2 | 14 | 13.0 (13.0) | 13.0 | 154 / 154 | 81 | 13 | 0.0 (0.0) | 0.0 | 158 / 158 | 85 | Y |
| batch000_e1 | 3 | 11 | 1.9 (1.9) | 2.0 | 138 / 138 | 47 | 11 | 0.0 (0.0) | 0.0 | 144 / 144 | 59 | Y |
| mayonnaise_jar_asset_0008 | 0 | 13 | 1.0 (1.1) | 0.8 | 156 / 156 | 24 | 12 | 3.9 (10.0) | 2.0 | 148 / 148 | 10 | Y |
| mayonnaise_jar_asset_0008 | 1 | 12 | 6.1 (6.1) | 5.9 | 144 / 144 | 13 | 14 | 3.9 (9.9) | 2.0 | 123 / 123 | 53 | Y |
| mayonnaise_jar_asset_0008 | 2 | 12 | 50.1 (50.1) | 35.5 | 144 / 144 | 8 | 9 | 1.4 (9.9) | 0.7 | 139 / 139 | 32 | Y |
| mayonnaise_jar_asset_0008 | 3 | 17 | 49.7 (49.7) | 40.7 | 155 / 155 | 16 | 14 | 3.9 (10.0) | 2.0 | 154 / 154 | 9 | Y |
| spice_jar_asset_0025 | 0 | 23 | 16.1 (16.1) | 14.7 | 129 / 129 | 33 | 16 | 0.1 (0.1) | 0.0 | 149 / 149 | 33 | Y |
| spice_jar_asset_0025 | 1 | 22 | 17.9 (17.9) | 15.4 | 149 / 117 | 33 | 20 | 0.1 (0.4) | 0.0 | 141 / 117 | 39 | Y |
| spice_jar_asset_0025 | 2 | 20 | 23.4 (23.4) | 21.6 | 127 / 127 | 24 | 13 | 0.1 (0.1) | 0.0 | 123 / 123 | 37 | Y |
| spice_jar_asset_0025 | 3 | 25 | 14.2 (14.3) | 19.1 | 139 / 139 | 45 | 19 | 0.1 (0.1) | 0.0 | 155 / 155 | 55 | Y |
| cracker_box_asset_0039 | 0 | 17 | 4.8 (4.8) | 4.9 | 148 / 148 | 5 | - | 0.3 (2.5) | 0.3 | 0 / 0 | 213 | n |
| cracker_box_asset_0039 | 1 | 19 | 13.3 (13.3) | 16.2 | 158 / 158 | 19 | - | 0.0 (0.7) | 0.0 | 0 / -0 | 213 | n |
| cracker_box_asset_0039 | 2 | 20 | 24.8 (24.8) | 25.5 | 183 / 183 | 34 | 20 | 8.5 (8.5) | 10.8 | 220 / 220 | 82 | Y |
| cracker_box_asset_0039 | 3 | 19 | 17.1 (17.1) | 20.8 | 148 / 148 | 18 | 20 | 14.3 (14.3) | 19.5 | 6 / -60 | 224 | n |
| cereal_box_asset_0044 | 0 | 25 | 23.1 (23.1) | 24.4 | 145 / 145 | 17 | - | 85.2 (85.2) | 102.6 | 0 / -66 | 231 | n |
| cereal_box_asset_0044 | 1 | 14 | 7.6 (7.6) | 8.2 | 186 / 186 | 48 | 7 | 7.7 (7.7) | 9.9 | 172 / 172 | 68 | Y |
| cereal_box_asset_0044 | 2 | 22 | 25.5 (25.5) | 24.6 | 190 / 190 | 48 | - | 85.3 (85.9) | 104.1 | 0 / -66 | 231 | n |
| cereal_box_asset_0044 | 3 | 20 | 18.7 (18.7) | 20.4 | 130 / 130 | 29 | - | 85.3 (86.0) | 104.2 | 0 / -66 | 231 | n |
| tea_box_asset_0019 | 0 | 16 | 5.4 (5.4) | 7.5 | 152 / 152 | 15 | 15 | 0.0 (0.0) | 0.0 | 12 / -32 | 252 | n |
| tea_box_asset_0019 | 1 | 8 | 0.1 (0.1) | 0.2 | 210 / 146 | 5 | 11 | 18.4 (18.4) | 14.9 | 10 / -32 | 231 | n |
| tea_box_asset_0019 | 2 | 10 | 0.9 (0.9) | 0.8 | 253 / 253 | 126 | 10 | 0.0 (0.0) | 0.0 | 6 / -0 | 214 | n |
| tea_box_asset_0019 | 3 | 13 | 0.6 (0.6) | 0.6 | 144 / 144 | 11 | 13 | 0.0 (0.0) | 0.0 | 6 / -32 | 249 | n |
| pickle_jar_asset_0025 | 0 | 11 | 0.5 (0.5) | 0.2 | 179 / 179 | 38 | 11 | 1.6 (2.8) | 1.9 | 6 / -26 | 183 | n |
| pickle_jar_asset_0025 | 1 | 11 | 0.6 (0.6) | 0.7 | 173 / 173 | 145 | 9 | 1.6 (2.8) | 1.9 | 4 / -0 | 201 | n |
| pickle_jar_asset_0025 | 2 | 10 | 0.7 (0.7) | 0.6 | 167 / 167 | 26 | 9 | 1.6 (2.8) | 1.9 | 4 / -28 | 211 | n |
| pickle_jar_asset_0025 | 3 | 11 | 4.4 (4.4) | 5.2 | 181 / 181 | 36 | 10 | 1.6 (2.8) | 1.9 | 5 / -27 | 224 | n |
| pasta_box_asset_0058 | 0 | 23 | 24.2 (24.2) | 23.5 | 168 / 168 | 30 | - | - | - | - | - | - |
| pasta_box_asset_0058 | 1 | 15 | 27.6 (27.6) | 23.4 | 152 / 152 | 49 | - | - | - | - | - | - |
| pasta_box_asset_0058 | 2 | 12 | 15.4 (16.2) | 14.1 | 132 / 132 | 55 | - | - | - | - | - | - |
| pasta_box_asset_0058 | 3 | 18 | 17.1 (17.1) | 16.1 | 179 / 179 | 38 | - | - | - | - | - | - |

## Recorded clips (gt)

| episode | contact f | tilt@c (max) deg | disp@c mm | peak/final rise mm |
|---|---|---|---|---|
| peanut_butter_jar_asset_0003 | 14 | 0.0 (0.0) | 1.3 | 206 / 206 |
| jam_jar_asset_0085 | 14 | 1.8 (3.5) | 2.9 | 357 / 357 |
| batch000_e1 | 15 | 0.0 (0.0) | 2.4 | 273 / 273 |
| mayonnaise_jar_asset_0008 | 14 | 4.0 (4.0) | 4.1 | 264 / 264 |
| spice_jar_asset_0025 | 19 | 1.3 (1.3) | 1.5 | 257 / 257 |
| cracker_box_asset_0039 | 28 | 6.9 (6.9) | 9.2 | 185 / 185 |
| cereal_box_asset_0044 | 25 | 0.2 (0.6) | 1.4 | 210 / 210 |
| tea_box_asset_0019 | 14 | 0.0 (0.0) | 0.8 | 316 / 316 |
| pickle_jar_asset_0025 | 16 | 0.0 (0.2) | 0.7 | 176 / 176 |
| pasta_box_asset_0058 | 24 | 13.5 (18.2) | 10.6 | 225 / 225 |

Failed / incomplete: pasta_box_asset_0058, juice_carton_asset_0025, milk_carton_asset_0037, food_can_asset_0039, soda_can_asset_0012, tuna_can_asset_0023, mustard_bottle_asset_0059, water_bottle_asset_0087, drinking_glass_asset_0056, lemon_asset_0012, tomato_asset_0012

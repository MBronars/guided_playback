# stage2: letter moulding arms (mould ctxgen step 60k, 3 members, episode 0)

imagined = best all-frame objective per member (RMS mm); last = last-frame chamfer of the picked member (imagined); sim = Genesis last-frame chamfer (paper metric), picked member and best member; unguided = last-frame chamfer of the unguided draws.

| arm | knobs | picked | imagined obj | imagined last | sim picked | sim best (m) | unguided last | gap | guide s |
|---|---|---|---|---|---|---|---|---|---|
| letters2_R_last15_it48_lr0.1 | R it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m2 | 4.83 | 4.58 | 5.52 | 5.51 (m0) | 7.71 | 0.94 | 469.9 |
| letters2_C_last15_it24_lr0.1 | C it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m1 | 4.95 | 4.92 | 5.71 | 5.71 (m1) | 14.41 | 1.63 | 297.8 |
| letters2_L_last15_it24_lr0.1 | L it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m1 | 5.87 | 5.87 | 6.22 | 6.22 (m1) | 13.83 | 2.38 | 297.7 |
| letters2_I_last15_it24_lr0.1 | I it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m1 | 8.26 | 8.49 | 10.74 | 9.13 (m0) | 17.33 | 2.7 | 298.6 |
| letters2_C_last15_it48_ph1 | C it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m0 | 6.20 | 5.75 | 5.45 | 5.45 (m0) | 14.41 | 0.94 | 531.2 |
| letters2_L_last15_it48_ph1 | L it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m2 | 7.52 | 6.86 | 6.98 | 5.96 (m0) | 13.83 | 1.77 | 530.8 |
| letters2_R_last15_it48_ph1 | R it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m0 | 5.84 | 5.34 | 5.19 | 5.19 (m0) | 7.71 | 0.68 | 530.6 |
| letters2_C_all_it48_ph1 | C it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m0 | 7.65 | 4.20 | 3.55 | 3.55 (m0) | 14.41 | 1.85 | 534.3 |
| letters2_I_last15_it48_ph1 | I it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m2 | 6.19 | 5.05 | 9.18 | 9.18 (m2) | 17.33 | 5.65 | 532.9 |
| letters2_C_last15_it48_ph3 | C it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m2 | 5.99 | 4.30 | 5.59 | 5.59 (m2) | 14.41 | 1.55 | 530.5 |
| letters2_R_last15_it48_lr0.2to0.02 | R it48 lr 0.2->0.02 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m2 | 5.03 | 5.00 | 5.45 | 5.45 (m2) | 7.71 | 1.99 | 505.4 |
| letters2_R_last15_it24_lr0.2 | R it24 lr 0.2 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m1 | 5.00 | 5.00 | 5.48 | 5.48 (m1) | 7.71 | 1.56 | 297.8 |
| letters2_R_last15_it24_seed1 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 1 | m0 | 5.01 | 5.35 | 6.29 | 6.29 (m0) | 13.21 | 2.56 | 298.3 |
| letters2_R_last15_it24_seed2 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 2 | m2 | 4.24 | 4.33 | 5.22 | 5.22 (m2) | 7.69 | 1.61 | 298.2 |
| letters2_R_last1_it48_lr0.1 | R it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 1 seed 0 | m2 | 4.97 | 4.97 | 5.57 | 5.28 (m0) | 7.71 | 0.6 | 530.2 |
| letters2_C_last15_it48_lr0.1 | C it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m2 | 5.75 | 5.86 | 7.22 | 7.22 (m2) | 14.41 | 4.33 | 517.8 |
| letters2_L_last15_it48_lr0.1 | L it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m1 | 5.62 | 5.62 | 7.96 | 7.96 (m1) | 13.83 | 5.94 | 458.0 |
| letters2_I_last15_it48_lr0.1 | I it48 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m2 | 6.26 | 6.21 | 7.92 | 7.92 (m2) | 17.33 | 3.47 | 464.0 |
| letters2_R_last15_it96_lr0.1 | R it96 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m0 | 4.52 | 4.52 | 4.84 | 4.84 (m0) | 7.71 | 2.64 | 918.3 |
| letters2_R_last15_it48_S10 | R it48 lr 0.1 unitgrad S10 tf0.8 N150 frames 15 seed 0 | m0 | 5.69 | 5.68 |  |  () | 6.57 |  | 761.5 |

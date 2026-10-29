# stage3: letter moulding arms (mould ctxgen step 60k, 3 members, episode 0)

imagined = best all-frame objective per member (RMS mm); last = last-frame chamfer of the picked member (imagined); sim = Genesis last-frame chamfer (paper metric), picked member and best member; unguided = last-frame chamfer of the unguided draws.

| arm | knobs | picked | imagined obj | imagined last | sim picked | sim best (m) | unguided last | gap | guide s |
|---|---|---|---|---|---|---|---|---|---|
| letters_R_it24_last1 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 1 seed 0 | m2 | 6.03 | 6.03 | 5.4 | 5.4 (m2) | 7.71 | 1.95 | 231.4 |
| letters_I_it48_lr0.1 | I it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m1 | 7.93 | 5.74 | 10.02 | 9.37 (m0) | 17.33 | 4.28 | 459.8 |
| letters3_R_all_it48_ph1 | R it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m0 | 6.40 | 6.40 | 6.37 | 6.31 (m2) | 7.71 | 1.72 | 466.4 |
| letters3_L_all_it48_ph1 | L it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m0 | 9.69 | 9.52 | 12.81 | 10.44 (m2) | 13.83 | 5.54 | 461.7 |
| letters3_I_all_it48_ph1 | I it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m0 | 9.78 | 6.18 | 7.86 | 7.86 (m0) | 17.33 | 7.79 | 462.9 |
| letters3_C_all_it48_ph1_seed1 | C it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 1 | m0 | 8.47 | 7.16 | 9.49 | 5.33 (m1) | 12.18 | 4.42 | 461.6 |
| letters3_R_last15_S10_ph1_it24 | R it24 lr 0.1 unitgrad S10 tf0.8 N150 frames 15 seed 0 | m1 | 6.86 | 6.61 | 6.04 | 6.04 (m1) | 6.57 | 1.84 | 383.2 |
| letters3_I_last15_S10_ph1_it24 | I it24 lr 0.1 unitgrad S10 tf0.8 N150 frames 15 seed 0 | m1 | 8.63 | 7.49 | 10.08 | 10.08 (m1) | 15.86 | 2.6 | 383.2 |
| letters3_R_last15_T200_ph1_it24 | R it24 lr 0.1 unitgrad S6 tf0.8 N200 frames 15 seed 0 | m0 | 6.79 | 7.03 | 6.89 | 6.89 (m0) | 9.59 | 1.16 | 302.4 |
| letters3_I_last15_T200_ph1_it24 | I it24 lr 0.1 unitgrad S6 tf0.8 N200 frames 15 seed 0 | m2 | 9.59 | 8.79 | 13.3 | 9.83 (m0) | 15.95 | 4.51 | 303.6 |
| letters3_C_all_it48_ph1_seed2 | C it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 2 | m2 | 9.03 | 8.08 | 14.34 | 10.43 (m1) | 11.80 | 6.26 | 460.5 |
| letters3_L_last15_ph1_it24_seed1 | L it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 1 | m0 | 10.25 | 9.13 | 10.76 | 10.35 (m2) | 18.44 | 1.63 | 230.7 |
| letters3_L_last15_ph1_it24_seed2 | L it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 2 | m0 | 6.23 | 5.23 | 7.47 | 6.6 (m1) | 19.35 | 2.24 | 231.1 |
| letters3_R_last15_ph1_it24_seed3 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 3 | m0 | 5.32 | 5.00 | 6.0 | 6.0 (m0) | 7.31 | 1.43 | 232.3 |
| letters3_R_last15_ph1_it24_seed4 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 4 | m1 | 6.47 | 5.40 | 5.25 | 5.25 (m1) | 10.37 | 1.71 | 232.6 |
| letters3_R_last15_ph1_it24_seed5 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 5 | m2 | 6.09 | 5.78 | 5.77 | 5.77 (m2) | 7.95 | 0.42 | 232.4 |
| letters3_C_all_it96_ph1 | C it96 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m0 | 6.79 | 4.61 | 5.92 | 4.77 (m2) | 14.41 | 1.54 | 916.9 |
| letters3_R_all_S10_T200_ph1_it48 | R it48 lr 0.1 unitgrad S10 tf0.8 N200 frames all seed 0 | m2 | 6.41 | 5.79 | 7.09 | 7.09 (m2) | 7.90 | 5.11 | 997.9 |

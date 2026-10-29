# stage1: letter moulding arms (mould ctxgen step 60k, 3 members, episode 0)

imagined = best all-frame objective per member (RMS mm); last = last-frame chamfer of the picked member (imagined); sim = Genesis last-frame chamfer (paper metric), picked member and best member; unguided = last-frame chamfer of the unguided draws.

| arm | knobs | picked | imagined obj | imagined last | sim picked | sim best (m) | unguided last | gap | guide s |
|---|---|---|---|---|---|---|---|---|---|
| letters_R_it24_lr0.1 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 6.10 | 6.46 | 8.18 | 6.6 (m0) | 7.71 | 1.73 | 237.6 |
| letters_C_it24_lr0.1 | C it24 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m0 | 9.80 | 9.27 | 8.14 | 5.14 (m2) | 14.41 | 1.13 | 300.9 |
| letters_L_it24_lr0.1 | L it24 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m0 | 9.28 | 9.29 | 10.42 | 9.51 (m1) | 13.83 | 4.05 | 300.9 |
| letters_I_it24_lr0.1 | I it24 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 9.77 | 6.25 | 8.62 | 8.62 (m2) | 17.33 | 6.16 | 303.6 |
| letters_R_it48_lr0.1 | R it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 5.71 | 5.55 | 6.43 | 6.38 (m1) | 7.71 | 0.88 | 540.8 |
| letters_R_it24_lr0.05 | R it24 lr 0.05 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 5.49 | 5.12 | 6.97 | 6.28 (m0) | 7.71 | 1.84 | 302.3 |
| letters_R_it24_lr0.2 | R it24 lr 0.2 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 5.67 | 5.35 | 6.74 | 5.03 (m0) | 7.71 | 1.39 | 304.4 |
| letters_R_it48_lr0.2to0.02 | R it48 lr 0.2->0.02 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 5.34 | 4.87 | 6.49 | 6.49 (m2) | 7.71 | 2.01 | 544.0 |
| letters_R_it24_tf0.5 | R it24 lr 0.1 unitgrad S6 tf0.5 N150 frames all seed 0 | m0 | 7.64 | 7.71 | 7.16 | 7.16 (m0) | 7.71 | 12.5 | 304.6 |
| letters_R_it24_tf0.65 | R it24 lr 0.1 unitgrad S6 tf0.65 N150 frames all seed 0 | m2 | 6.44 | 6.71 | 9.74 | 6.74 (m0) | 7.71 | 8.55 | 303.1 |
| letters_R_it24_last15 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 15 seed 0 | m1 | 4.82 | 4.81 | 5.03 | 5.03 (m1) | 7.71 | 0.95 | 302.8 |
| letters_R_it24_last1 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames 1 seed 0 | m1 | 4.75 | 4.75 | 5.06 | 5.06 (m1) | 7.71 | 0.93 | 301.5 |
| letters_C_it48_lr0.1 | C it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m0 | 5.74 | 4.94 | 18.69 | 14.14 (m2) | 14.41 | 13.75 | 543.3 |
| letters_L_it48_lr0.1 | L it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 9.11 | 5.30 | 15.84 | 11.38 (m0) | 13.83 | 10.54 | 542.6 |
| letters_I_it48_lr0.1 | I it48 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 9.40 | 5.87 | 11.45 | 8.37 (m0) | 17.33 | 5.57 | 541.9 |
| letters_R_it24_S10 | R it24 lr 0.1 unitgrad S10 tf0.8 N150 frames all seed 0 | m1 | 5.14 | 4.96 | 5.44 | 5.44 (m1) | 6.57 | 0.77 | 464.9 |
| letters_R_it24_adam0.1 | R it24 lr 0.1 adam S6 tf0.8 N150 frames all seed 0 | m2 | 5.85 | 5.49 | 6.65 | 5.34 (m0) | 7.71 | 1.16 | 304.7 |
| letters_R_it24_lr0.1_seed1 | R it24 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 1 | m0 | 6.42 | 9.36 | 11.12 | 7.22 (m1) | 13.21 | 1.76 | 303.9 |
| letters_R_it96_lr0.1 | R it96 lr 0.1 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 5.25 | 4.49 | 6.37 | 6.24 (m0) | 7.71 | 1.88 | 1023.8 |
| letters_R_it48_lr0.1to0.01 | R it48 lr 0.1->0.01 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 5.41 | 6.62 | 7.86 | 6.12 (m0) | 7.71 | 1.81 | 536.8 |
| letters_R_it24_T200 | R it24 lr 0.1 unitgrad S6 tf0.8 N200 frames all seed 0 | m1 | 5.58 | 5.29 | 5.66 | 5.66 (m1) | 9.59 | 0.57 | 373.0 |
| letters_C_it48_lr0.2to0.02 | C it48 lr 0.2->0.02 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 5.66 | 3.51 | 16.31 | 15.76 (m0) | 14.41 | 12.79 | 551.9 |
| letters_L_it48_lr0.2to0.02 | L it48 lr 0.2->0.02 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 8.87 | 5.79 | 15.67 | 5.31 (m1) | 13.83 | 9.88 | 531.7 |
| letters_I_it48_lr0.2to0.02 | I it48 lr 0.2->0.02 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 3.64 | 3.22 | 10.18 | 9.41 (m0) | 17.33 | 6.96 | 528.7 |
| letters_C_it24_tf0.5 | C it24 lr 0.1 unitgrad S6 tf0.5 N150 frames all seed 0 | m0 | 13.54 | 9.86 | 22.25 | 12.15 (m2) | 14.41 | 12.4 | 298.7 |
| letters_L_it24_tf0.5 | L it24 lr 0.1 unitgrad S6 tf0.5 N150 frames all seed 0 | m1 | 14.43 | 13.35 | 14.72 | 14.72 (m1) | 13.83 | 3.08 | 298.8 |
| letters_I_it24_tf0.5 | I it24 lr 0.1 unitgrad S6 tf0.5 N150 frames all seed 0 | m0 | 10.27 | 10.26 | 13.37 | 12.06 (m1) | 17.33 | 26.91 | 298.0 |
| letters_R_it96_lr0.2to0.02 | R it96 lr 0.2->0.02 unitgrad S6 tf0.8 N150 frames all seed 0 | m2 | 3.84 | 3.27 | 14.81 | 5.67 (m0) | 7.71 | 11.54 | 998.0 |

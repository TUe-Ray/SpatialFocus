# COCO Coverage Report for VLM-3R Selected Words

Dataset analyzed:
- `/leonardo/home/userexternal/shuang00/Word-Selection_LLM/artifacts/vlm3r_word_selection/vsibench_split`

Comparison target:
- `COCO_PANOPTIC_CLASS_NAMES`

## What “before” and “after” mean

- **Before patch**: direct class-name matching only (`exact`)
- **After patch**: current runtime matcher (`hybrid_safe`)

Current note:
- On the full artifact vocabulary, `hybrid_safe` produced the same aggregate coverage as `exact_alias`.
- That means the gain currently comes from normalization + curated alias mapping, not from extra fuzzy recoveries at dataset scale.

## Global results

### `visible_grounded_words`

| Metric | Before (`exact`) | After (`hybrid_safe`) | Delta |
|---|---:|---:|---:|
| Unique words matched | 42 / 335 | 58 / 335 | +16 |
| Unique coverage | 12.54% | 17.31% | +4.78 pp |
| Occurrence coverage | 267,970 / 608,379 | 366,516 / 608,379 | +98,546 |
| Occurrence coverage % | 44.05% | 60.24% | +16.20 pp |

### `selected_words`

| Metric | Before (`exact`) | After (`hybrid_safe`) | Delta |
|---|---:|---:|---:|
| Unique words matched | 42 / 425 | 58 / 425 | +16 |
| Unique coverage | 9.88% | 13.65% | +3.76 pp |
| Occurrence coverage | 267,970 / 1,381,392 | 366,516 / 1,381,392 | +98,546 |
| Occurrence coverage % | 19.40% | 26.53% | +7.13 pp |

## Per-split results

### `merged_qa_route_plan_train`

#### `visible_grounded_words`
- Before: 42 / 327 unique = 12.84%
- After: 56 / 327 unique = 17.13%
- Before occurrences: 4,684 / 12,301 = 38.08%
- After occurrences: 6,773 / 12,301 = 55.06%

#### `selected_words`
- Before: 42 / 347 unique = 12.10%
- After: 56 / 347 unique = 16.14%
- Before occurrences: 4,684 / 35,318 = 13.26%
- After occurrences: 6,773 / 35,318 = 19.18%

### `merged_qa_scannet_train`

#### `visible_grounded_words`
- Before: 16 / 34 unique = 47.06%
- After: 22 / 34 unique = 64.71%
- Before occurrences: 87,012 / 154,727 = 56.24%
- After occurrences: 109,814 / 154,727 = 70.97%

#### `selected_words`
- Before: 16 / 79 unique = 20.25%
- After: 22 / 79 unique = 27.85%
- Before occurrences: 87,012 / 350,786 = 24.80%
- After occurrences: 109,814 / 350,786 = 31.31%

### `merged_qa_scannetpp_train`

#### `visible_grounded_words`
- Before: 24 / 85 unique = 28.24%
- After: 36 / 85 unique = 42.35%
- Before occurrences: 176,274 / 441,351 = 39.94%
- After occurrences: 249,929 / 441,351 = 56.63%

#### `selected_words`
- Before: 24 / 155 unique = 15.48%
- After: 36 / 155 unique = 23.23%
- Before occurrences: 176,274 / 995,288 = 17.71%
- After occurrences: 249,929 / 995,288 = 25.11%

## Interpretation

- The patch improves real coverage meaningfully, especially for `visible_grounded_words`, which is the intended runtime source.
- The gain is helpful but still limited by the COCO taxonomy.
- Many frequent indoor words still have no reliable COCO class target, for example:
  - `light switch`
  - `smoke detector`
  - `heater`
  - `cabinet`
  - `bookshelf`
  - `whiteboard`
  - `radiator`
  - `desk`

## What the patch is actually buying

High-value recovered mappings include:
- `sofa -> couch`
- `table -> dining_table`
- `monitor -> tv`
- `office chair -> chair`
- `blinds -> window_blind`
- `telephone -> cell_phone`
- `lamp -> light`
- `ceiling lamp -> light`
- `table lamp -> light`
- `plant -> potted_plant`

## Bottom line

- The patch makes word-aware filtering materially safer and more useful.
- It does **not** solve the long-tail indoor taxonomy gap.
- That is why the runtime default remains:
  - `visible_grounded_words`
  - `hybrid_safe`
  - `keep_masks` on no-match

## Remaining unmatched `visible_grounded_words` by split

These are the `visible_grounded_words` that still do **not** map to a COCO panoptic class after the current matcher (`hybrid_safe`), with occurrence counts from each split's `selected_words.jsonl`.

### `merged_qa_route_plan_train`

Unmatched unique words: `271`

`cabinet` (519), `stool` (331), `trash can` (244), `doorframe` (242), `fireplace` (182), `dishwasher` (177), `whiteboard` (176), `smoke detector` (162), `desk` (156), `bathtub` (143), `washer` (140), `coffee table` (124), `radiator` (107), `bookshelf` (97), `floor` (97), `jacket` (96), `light switch` (84), `kitchen counter` (77), `box` (76), `dresser` (69), `picture` (59), `tv stand` (59), `paper towel` (55), `copier` (49), `heater` (49), `bathroom vanity` (47), `shoes` (45), `recycling bin` (44), `basket` (43), `board` (43), `armchair` (42), `ottoman` (40), `shower curtain` (38), `mat` (36), `bathroom cabinet` (34), `paper bag` (33), `computer tower` (31), `ceiling light` (30), `nightstand` (28), `paper towel dispenser` (26), `sofa chair` (25), `closet doors` (24), `laundry basket` (23), `bucket` (22), `printer` (22), `cabinets` (21), `file cabinet` (20), `headphones` (20), `scale` (20), `mini fridge` (19), `shower` (19), `clothing` (18), `fan` (18), `shower walls` (18), `closet` (17), `guitar` (17), `piano` (17), `spray bottle` (17), `kitchen cabinet` (16), `stand` (16), `slippers` (15), `storage cabinet` (15), `structure` (15), `clothes` (14), `wardrobe closet` (14), `ball` (13), `pan` (13), `breakfast bar` (12), `build_in_cabinet` (12), `case` (12), `ceiling fan` (12), `coat rack` (12), `staircase` (12), `step stool` (12), `tissue box` (12), `bathroom stall door` (11), `bulletin board` (11), `decoration` (11), `dumbbell` (11), `end table` (11), `footrest` (11), `furniture` (11), `kettle` (11), `paper cutter` (11), `purse` (11), `washing machine` (11), `boiler` (10), `closet ceiling` (10), `foosball table` (10), `kitchen cabinets` (10), `laundry hamper` (10), `shoe rack` (10), `shower curtain rod` (10), `soap dispenser` (10), `washing machines` (10), `water cooler` (10), `seat` (9), `stapler` (9), `vent` (9), `bathroom stall` (8), `blackboard` (8), `carpet` (8), `file folder` (8), `flowerpot` (8), `kitchen island` (8), `sign` (8), `stair rail` (8), `toilet paper` (8), `trash bin` (8), `wardrobe cabinet` (8), `lamp base` (7), `mailboxes` (7), `piano bench` (7), `poster` (7), `rug` (7), `storage bin` (7), `toilet paper dispenser` (7), `vacuum cleaner` (7), `bathroom counter` (6), `binder` (6), `bookshelves` (6), `closet door` (6), `container` (6), `cushion` (6), `fire extinguisher` (6), `ledge` (6), `rack` (6), `water fountain` (6), `boxes` (5), `boxes of paper` (5), `briefcase` (5), `chest` (5), `coat hanger` (5), `divider` (5), `folded chair` (5), `hamper` (5), `hoverboard` (5), `ikea bag` (5), `ironing board` (5), `ladder` (5), `machine` (5), `messenger bag` (5), `music stand` (5), `ping pong table` (5), `round table` (5), `sofa bed` (5), `storage box` (5), `alarm` (4), `bin` (4), `cabinet doors` (4), `calendar` (4), `closet wall` (4), `closet walls` (4), `clothes dryer` (4), `column` (4), `crate` (4), `futon` (4), `globe` (4), `handicap bar` (4), `hose` (4), `luggage` (4), `pool table` (4), `pot` (4), `power strip` (4), `projector screen` (4), `storage organizer` (4), `toilet paper holder` (4), `toilet seat cover dispenser` (4), `water bottle` (4), `windowsill` (4), `blackboard eraser` (3), `cart` (3), `closet rod` (3), `coat` (3), `drawer` (3), `elevator` (3), `fire alarm` (3), `hand dryer` (3), `magazine rack` (3), `oven mitt` (3), `pillar` (3), `plate` (3), `podium` (3), `power outlet` (3), `projector` (3), `rice cooker` (3), `shower floor` (3), `shredder` (3), `stack of chairs` (3), `swiffer` (3), `water bottles` (3), `wheel` (3), `banister` (2), `barricade` (2), `barrier` (2), `bath walls` (2), `case of water bottles` (2), `changing station` (2), `cloth` (2), `desk lamp` (2), `dish rack` (2), `dustpan` (2), `elevator button` (2), `faucet` (2), `fire hose` (2), `folded chairs` (2), `footstool` (2), `hat` (2), `jar` (2), `lunch box` (2), `magazine` (2), `media center` (2), `mirror doors` (2), `mop` (2), `mug` (2), `quadcopter` (2), `rail` (2), `range hood` (2), `shoe` (2), `shower wall` (2), `sweater` (2), `tray` (2), `alarm clock` (1), `beachball` (1), `cabinet door` (1), `chandelier` (1), `coffee maker` (1), `couch cushions` (1), `dolly` (1), `drum set` (1), `dumbbell plates` (1), `elliptical machine` (1), `exhaust fan` (1), `exit sign` (1), `flip flops` (1), `folded ladder` (1), `garage door` (1), `grocery bag` (1), `hand towel` (1), `iron` (1), `keyboard piano` (1), `loft bed` (1), `mail` (1), `nerf gun` (1), `organizer` (1), `painting` (1), `pictures` (1), `pipes` (1), `plastic bin` (1), `railing` (1), `rod` (1), `scanner` (1), `screen` (1), `shopping bag` (1), `shorts` (1), `shower door` (1), `slipper` (1), `stacks of cups` (1), `statue` (1), `step` (1), `stuffed animal` (1), `suitcases` (1), `switch` (1), `telescope` (1), `trash cabinet` (1), `trunk` (1), `tube` (1), `wall hanging` (1), `water pitcher` (1), `wet floor sign` (1), `whiteboard eraser` (1)

### `merged_qa_scannet_train`

Unmatched unique words: `12`

`trash bin` (12769), `radiator` (7913), `desk` (6682), `nightstand` (4693), `bookshelf` (4658), `fan` (2059), `closet` (1579), `room` (1475), `printer` (1401), `guitar` (1026), `piano` (384), `washing machine` (274)

### `merged_qa_scannetpp_train`

Unmatched unique words: `49`

`light switch` (17963), `smoke detector` (17309), `heater` (17225), `trash can` (12957), `cabinet` (11213), `whiteboard` (8746), `box` (7433), `power strip` (5651), `jacket` (4895), `bookshelf` (4750), `picture` (4738), `shoes` (3758), `poster` (3732), `storage cabinet` (3644), `basket` (3601), `whiteboard eraser` (3374), `kettle` (3348), `bucket` (3179), `toilet paper` (3148), `headphones` (2887), `paper towel` (2804), `toilet brush` (2773), `speaker` (2700), `slippers` (2618), `paper bag` (2571), `soap dispenser` (2547), `cutting board` (2477), `computer tower` (2362), `spray bottle` (2122), `printer` (1967), `coat hanger` (1863), `pot` (1839), `tissue box` (1784), `jar` (1758), `painting` (1721), `kitchen cabinet` (1685), `room` (1558), `socket` (1447), `shoe rack` (1374), `exhaust fan` (1149), `stapler` (1080), `pan` (1075), `container` (1018), `cushion` (990), `crate` (859), `rack` (596), `file folder` (587), `binder` (314), `marker` (233)

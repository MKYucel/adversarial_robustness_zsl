from torch.utils.data import Dataset
from PIL import Image
import os
import torch

# Class names are incorrectly parsed, it won't affect the results.

label_map = {0 :  "abbey", 1 :  "access_road", 2 :  "airfield", 3 :  "airlock", 4 :  "airplane_cabin", 5 :  "airport",
6 :  "entrance", 7 :  "airport_terminal", 8 :  "airport_ticket_counter", 9 :  "alcove", 10 :  "alley", 11 :  "amphitheater",
12 :  "amusement_arcade", 13 :  "amusement_park", 14 :  "anechoic_chamber", 15 :  "outdoor", 16 :  "indoor",
17 :  "outdoor", 18 :  "aquarium", 19 :  "aquatic_theater", 20 :  "aqueduct", 21 :  "arch", 22 :  "archaelogical_excavation",
23 :  "archive", 24 :  "basketball", 25 :  "hockey", 26 :  "performance", 27 :  "armory", 28 :  "outdoor",
29 :  "art_gallery", 30 :  "art_school", 31 :  "art_studio", 32 :  "artists_loft", 33 :  "assembly_line",
34 :  "outdoor", 35 :  "home", 36 :  "public", 37 :  "attic", 38 :  "auditorium", 39 :  "auto_factory",
40 :  "indoor", 41 :  "auto_racing_paddock", 42 :  "auto_showroom", 43 :  "backstage", 44 :  "badlands",
45 :  "indoor", 46 :  "outdoor", 47 :  "baggage_claim", 48 :  "kitchen", 49 :  "shop", 50 :  "exterior", 51 :  "interior",
52 :  "ball_pit", 53 :  "ballroom", 54 :  "bamboo_forest", 55 :  "indoor", 56 :  "outdoor", 57 :  "bank_vault",
58 :  "banquet_hall", 59 :  "indoor", 60 :  "outdoor", 61 :  "bar", 62 :  "barn", 63 :  "barndoor",
64 :  "baseball_field", 65 :  "basement", 66 :  "basilica", 67 :  "indoor", 68 :  "outdoor", 69 :  "bathroom",
70 :  "batters_box", 71 :  "indoor", 72 :  "outdoor", 73 :  "bayou", 74 :  "indoor", 75 :  "outdoor", 76 :  "beach",
77: "beach_house", 78: "beauty_salon", 79: "bedchamber", 80: "bedroom", 81: "beer_garden", 82: "beer_hall",
83: "bell_foundry", 84: "berth", 85: "betting_shop", 86: "bicycle_racks", 87: "bindery",
88: "biology_laboratory", 89: "indoor", 90: "outdoor", 91: "outdoor", 92: "boardwalk", 93: "boat_deck",
94: "boathouse", 95: "bog", 96: "bookstore", 97: "indoor", 98: "botanical_garden", 99: "indoor",
100: "outdoor", 101: "bowling_alley", 102: "boxing_ring", 103: "indoor", 104: "outdoor", 105: "outdoor",
106: "bridge", 107: "building_complex", 108: "building_facade", 109: "bullpen", 110: "bullring",
111: "burial_chamber", 112: "outdoor", 113: "bus_interior", 114: "bus_shelter", 115: "outdoor",
116: "butchers_shop", 117: "butte", 118: "cabana", 119: "outdoor", 120: "cafeteria", 121: "call_center",
122: "campsite", 123: "campus", 124: "natural", 125: "urban", 126: "candy_store", 127: "canteen",
128: "canyon", 129: "backseat", 130: "frontseat", 131: "caravansary", 132: "cardroom", 133: "airplane",
134: "freestanding", 135: "outdoor", 136: "carrousel", 137: "indoor", 138: "outdoor", 139: "castle",
140: "catacomb", 141: "indoor", 142: "outdoor", 143: "catwalk", 144: "indoor", 145: "cemetery",
146: "chalet", 147: "chaparral", 148: "chapel", 149: "checkout_counter", 150: "cheese_factory",
151: "chemical_plant", 152: "chemistry_lab", 153: "indoor", 154: "outdoor", 155: "indoor", 156: "outdoor",
157: "childs_room", 158: "indoor", 159: "outdoor", 160: "indoor", 161: "outdoor", 162: "city",
163: "classroom", 164: "clean_room", 165: "cliff", 166: "indoor", 167: "outdoor", 168: "closet",
169: "clothing_store", 170: "coast", 171: "cockpit", 172: "coffee_shop", 173: "computer_room",
174: "conference_center", 175: "conference_hall", 176: "conference_room", 177: "confessional",
178: "construction_site", 179: "control_room", 180: "indoor", 181: "outdoor", 182: "indoor",
183: "corn_field", 184: "corral", 185: "corridor", 186: "cottage", 187: "cottage_garden",
188: "courthouse", 189: "courtroom", 190: "courtyard", 191: "exterior", 192: "crawl_space", 193: "creek",
194: "crevasse", 195: "crosswalk", 196: "office", 197: "cybercafe", 198: "dacha", 199: "indoor",
200: "dam", 201: "darkroom", 202: "day_care_center", 203: "delicatessen", 204: "dentists_office",
205: "departure_lounge", 207: "sand", 208: "vegetation", 206: "desert_road", 209: "indoor", 210: "outdoor",
 211: "home", 212: "vehicle", 213: "dining_car", 214: "dining_hall", 215: "dining_room", 216: "dirt_track",
 217: "discotheque", 218: "dock", 219: "dolmen", 220: "donjon", 221: "indoor", 222: "outdoor",
 223: "dorm_room", 224: "downtown", 225: "drainage_ditch", 226: "drill_rig", 227: "driveway",
 228: "outdoor", 229: "drugstore", 230: "dry_dock", 231: "dugout", 232: "earth_fissure",
 233: "editing_room", 234: "electrical_substation", 235: "door", 236: "freight_elevator", 237: "interior",
 238: "elevator_lobby", 239: "elevator_shaft", 240: "embassy", 241: "engine_room", 242: "indoor",
 243: "outdoor", 244: "estuary", 245: "excavation", 246: "exhibition_hall", 247: "indoor", 248: "outdoor",
 249: "fairway", 250: "farm", 251: "fastfood_restaurant", 252: "fence", 253: "outdoor", 254: "cultivated",
 256: "wild", 255: "field_road", 257: "fire_escape", 258: "fire_station", 259: "indoor", 260: "outdoor",
 261: "fish_farm", 262: "fishpond", 263: "fjord", 264: "indoor", 265: "outdoor", 266: "natural",
 267: "urban", 268: "flood", 269: "indoor", 270: "fly_bridge", 271: "food_court", 272: "football_field",
 273: "broadleaf", 274: "needleleaf", 275: "forest_path", 276: "forest_road", 277: "formal_garden",
 278: "fort", 279: "fortress", 280: "indoor", 281: "outdoor", 282: "fountain", 283: "freeway",
 284: "funeral_chapel", 285: "furnace_room", 286: "galley", 287: "game_room", 288: "gangplank",
 289: "indoor", 290: "outdoor", 291: "garbage_dump", 292: "gas_station", 293: "gasworks", 294: "gatehouse",
 295: "exterior", 296: "indoor", 297: "outdoor", 298: "indoor", 299: "outdoor", 300: "ghost_town",
 301: "gift_shop", 302: "glacier", 303: "golf_course", 304: "gorge", 305: "great_hall", 306: "indoor",
 307: "outdoor", 308: "grotto", 309: "guardhouse", 310: "gulch", 311: "indoor", 312: "indoor",
 313: "hacienda", 314: "hallway", 315: "indoor", 316: "outdoor", 317: "harbor", 318: "hayfield",
 319: "heath", 320: "hedge_maze", 321: "hedgerow", 322: "heliport", 323: "herb_garden", 324: "highway",
 325: "hill", 326: "home_office", 327: "home_theater", 328: "hoodoo", 329: "hospital", 330: "hospital_room",
 331: "hot_spring", 332: "indoor", 333: "outdoor", 335: "outdoor", 334: "hotel_breakfast_area",
 336: "hotel_room", 337: "house", 338: "indoor", 339: "outdoor", 340: "hut", 341: "ice_floe",
 342: "ice_shelf", 343: "indoor", 344: "outdoor", 345: "iceberg", 346: "igloo", 347: "industrial_area",
 348: "industrial_park", 349: "indoor", 350: "outdoor", 351: "irrigation_ditch", 352: "islet",
 353: "indoor", 354: "outdoor", 356: "indoor", 357: "outdoor", 355: "jail_cell", 358: "japanese_garden",
 359: "jewelry_shop", 360: "joss_house", 361: "junk_pile", 362: "junkyard", 363: "jury_box", 364: "kasbah",
 365: "indoor", 366: "outdoor", 367: "kindergarden_classroom", 368: "indoor", 369: "outdoor",
 370: "kitchen", 371: "kitchenette", 372: "lab_classroom", 373: "indoor", 374: "outdoor", 375: "lagoon",
 376: "artificial", 377: "natural", 378: "landfill", 379: "landing_deck", 380: "laundromat", 381: "lawn",
 382: "lean-to", 383: "lecture_room", 384: "levee", 385: "indoor", 386: "outdoor", 387: "outdoor",
 388: "lift_bridge", 389: "lighthouse", 390: "limousine_interior", 391: "indoor", 392: "outdoor",
 393: "living_room", 394: "loading_dock", 395: "lobby", 396: "lock_chamber", 397: "locker_room",
 398: "outdoor", 399: "machine_shop", 400: "manhole", 401: "mansion", 402: "manufactured_home",
 403: "indoor", 404: "outdoor", 405: "marsh", 406: "martial_arts_gym", 407: "mastaba", 408: "mausoleum",
 409: "medina", 410: "mesa", 411: "military_hospital", 412: "military_hut", 413: "mine", 414: "mineshaft",
 415: "outdoor", 416: "mission", 417: "dry", 418: "water", 419: "mobile_home", 420: "outdoor", 421: "moor",
 422: "morgue", 423: "indoor", 424: "outdoor", 425: "motel", 426: "mountain", 427: "mountain_path",
 428: "mountain_road", 429: "mountain_snowy", 430: "indoor", 431: "outdoor", 432: "indoor", 433: "outdoor",
 434: "music_store", 435: "music_studio", 436: "natural_history_museum", 437: "naval_base", 438: "newsroom",
 439: "outdoor", 440: "nightclub", 441: "indoor", 442: "outdoor", 443: "nursery", 444: "nursing_home",
 445: "oasis", 446: "oast_house", 447: "indoor", 448: "outdoor", 449: "ocean", 450: "office",
 451: "office_building", 452: "office_cubicles", 453: "outdoor", 454: "oilrig", 455: "operating_room",
 456: "optician", 457: "orchard", 458: "exterior", 459: "ossuary", 460: "outcropping", 461: "outdoor",
 462: "overpass", 463: "packaging_plant", 464: "pagoda", 465: "palace", 466: "pantry", 467: "parade_ground",
 468: "park", 469: "indoor", 470: "outdoor", 471: "parking_lot", 472: "parlor", 473: "particle_accelerator",
 474: "pasture", 475: "patio", 476: "pavilion", 477: "outdoor", 478: "pet_shop", 479: "pharmacy",
 480: "phone_booth", 481: "physics_laboratory", 482: "piano_store", 483: "picnic_area", 484: "pier",
 485: "pig_farm", 486: "indoor", 487: "outdoor", 488: "pitchers_mound", 489: "pizzeria", 490: "indoor",
 491: "outdoor", 492: "plantation_house", 493: "playground", 494: "playroom", 495: "plaza", 496: "indoor",
 497: "outdoor", 498: "pond", 499: "establishment", 500: "home", 501: "porch", 502: "portico",
 503: "indoor", 504: "outdoor", 505: "print_shop", 506: "priory", 507: "promenade", 508: "promenade_deck",
 509: "indoor", 510: "outdoor", 511: "pulpit", 512: "pump_room", 513: "putting_green", 514: "quadrangle",
 515: "quay", 516: "outdoor", 517: "racecourse", 518: "raceway", 519: "raft", 520: "railroad_track",
 521: "railway_yard", 522: "rainforest", 523: "ramp", 524: "ranch", 525: "ranch_house", 526: "reading_room",
 527: "reception", 528: "recreation_room", 529: "rectory", 530: "indoor", 531: "outdoor",
 532: "repair_shop", 533: "residential_neighborhood", 534: "resort", 535: "restaurant",
 536: "restaurant_kitchen", 537: "restaurant_patio", 538: "indoor", 539: "outdoor", 540: "revolving_door",
 541: "rice_paddy", 542: "riding_arena", 543: "river", 544: "road_cut", 545: "rock_arch",
 546: "rolling_mill", 547: "roof", 548: "root_cellar", 549: "rope_bridge", 550: "roundabout", 551: "rubble",
 552: "ruin", 553: "runway", 554: "sacristy", 555: "salt_plain", 556: "sand_trap", 557: "sandbar",
 558: "sandbox", 559: "sauna", 560: "savanna", 561: "sawmill", 562: "schoolhouse", 563: "schoolyard",
 564: "science_museum", 565: "sea_cliff", 566: "seawall", 567: "server_room", 568: "sewing_room",
 569: "shed", 570: "shipping_room", 571: "outdoor", 572: "shoe_shop", 573: "shopfront", 574: "indoor",
 575: "shower", 576: "signal_box", 577: "skatepark", 578: "ski_jump", 579: "ski_lodge", 580: "ski_resort",
 581: "ski_slope", 582: "sky", 583: "skyscraper", 584: "slum", 585: "snowfield", 586: "soccer_field",
 587: "spillway", 588: "squash_court", 589: "stable", 590: "baseball", 591: "football", 592: "outdoor",
 593: "soccer", 594: "indoor", 595: "outdoor", 596: "staircase", 597: "indoor", 598: "outdoor",
 599: "water", 600: "stone_circle", 601: "street", 602: "strip_mall", 603: "strip_mine",
 604: "submarine_interior", 605: "subway_interior", 606: "corridor", 607: "platform", 608: "sun_deck",
 609: "supermarket", 610: "sushi_bar", 611: "swamp", 612: "swimming_hole", 613: "indoor", 614: "outdoor",
 615: "indoor", 616: "outdoor", 617: "tea_garden", 618: "tearoom", 619: "teashop", 620: "television_studio",
 621: "east_asia", 622: "south_asia", 623: "western", 624: "indoor", 625: "outdoor", 626: "indoor",
 627: "outdoor", 628: "terrace_farm", 629: "indoor_procenium", 630: "indoor_round", 631: "indoor_seats",
 632: "outdoor", 633: "thriftshop", 634: "throne_room", 635: "ticket_booth", 636: "outdoor",
 637: "toll_plaza", 638: "tollbooth", 639: "topiary_garden", 640: "tower", 641: "town_house",
 642: "toyshop", 643: "indoor", 644: "outdoor", 645: "trading_floor", 646: "trailer_park",
 647: "train_depot", 648: "train_railway", 649: "outdoor", 650: "platform", 651: "station",
 652: "tree_farm", 653: "tree_house", 654: "trench", 655: "trestle_bridge", 656: "tundra",
 657: "rail_outdoor", 658: "road_outdoor", 659: "coral_reef", 660: "ice", 661: "kelp_forest",
 662: "ocean_deep", 663: "ocean_shallow", 664: "pool", 665: "wreck", 666: "utility_room", 667: "valley",
 668: "van_interior", 669: "vegetable_garden", 670: "indoor", 671: "outdoor", 672: "ventilation_shaft",
 673: "veranda", 674: "vestry", 675: "veterinarians_office", 676: "viaduct", 677: "videostore",
 678: "village", 679: "vineyard", 680: "volcano", 681: "outdoor", 682: "voting_booth", 683: "waiting_room",
 684: "indoor", 685: "watchtower", 686: "water_mill", 687: "water_tower", 688: "indoor", 689: "outdoor",
 690: "block", 691: "cascade", 692: "cataract", 693: "fan", 694: "plunge", 695: "watering_hole",
 696: "wave", 697: "weighbridge", 698: "wet_bar", 699: "wharf", 700: "wheat_field", 701: "wind_farm",
 702: "windmill", 703: "window_seat", 704: "barrel_storage", 705: "bottle_storage", 706: "winery",
 707: "witness_stand", 708: "woodland", 709: "workroom", 710: "workshop", 711: "indoor", 712: "yard",
 713: "youth_hostel", 714: "zen_garden", 715: "ziggurat", 716: "zoo"
 }

unseen_indices = [3,10,23,24,32,38,53,57,72,74,75,85,95,99,103,112,124,130
,138,145,152,158,184,196,216,221,237,245,246,254,259,262,286,298,315,328
,336,342,353,358,379,381,420,423,425,440,448,471,482,493,508,509,517,529
,558,560,580,622,631,635,645,650,656,658,674,679,681,695,710,711,712,715]

seen_indices = [x for x in range(len(label_map)) if x not in unseen_indices]


zsl_to_gzsl_label_indexes= {0: 3, 1: 10, 2: 23, 3: 24, 4: 32, 5: 38, 6: 53, 7: 57, 8: 72, 9: 74, 10: 75, 11: 85, 12: 95,
13: 99, 14: 103, 15: 112, 16: 124, 17: 130, 18: 138, 19: 145, 20: 152, 21: 158, 22: 184, 23: 196, 24: 216, 25: 221, 26: 237,
27: 245, 28: 246, 29: 254, 30: 259, 31: 262, 32: 286, 33: 298, 34: 315, 35: 328, 36: 336, 37: 342, 38: 353, 39: 358, 40: 379,
41: 381, 42: 420, 43: 423, 44: 425, 45: 440, 46: 448, 47: 471, 48: 482, 49: 493, 50: 508, 51: 509, 52: 517, 53: 529, 54: 558,
55: 560, 56: 580, 57: 622, 58: 631, 59: 635, 60: 645, 61: 650, 62: 656, 63: 658, 64: 674, 65: 679, 66: 681, 67: 695, 68: 710,
69: 711, 70: 712, 71: 715}

gzsl_to_zsl_label_indexes = {3: 0, 10: 1, 23: 2, 24: 3, 32: 4, 38: 5, 53: 6, 57: 7, 72: 8, 74: 9, 75: 10, 85: 11, 95: 12,
99: 13, 103: 14, 112: 15, 124: 16, 130: 17, 138: 18, 145: 19, 152: 20, 158: 21, 184: 22, 196: 23, 216: 24, 221: 25, 237: 26,
245: 27, 246: 28, 254: 29, 259: 30, 262: 31, 286: 32, 298: 33, 315: 34, 328: 35, 336: 36, 342: 37, 353: 38, 358: 39, 379:
40, 381: 41, 420: 42, 423: 43, 425: 44, 440: 45, 448: 46, 471: 47, 482: 48, 493: 49, 508: 50, 509: 51, 517: 52, 529: 53,
558: 54, 560: 55, 580: 56, 622: 57, 631: 58, 635: 59, 645: 60, 650: 61, 656: 62, 658: 63, 674: 64, 679: 65, 681: 66, 695: 67,
710: 68, 711: 69, 712: 70, 715: 71}




class SunDataset(Dataset):
    def __init__(self, indexes, files, labels , data_root, zsl = False,   transform=None):
        self.index_instances = indexes
        self.data_root = data_root

        self.file_names = files[self.index_instances - 1]
        self.labels = (labels[self.index_instances - 1] -1)

        if zsl:
            self.map_labels_zsl()

        self.transform = transform

    def __len__(self):
        return len(self.index_instances)

    def map_labels_zsl(self):
        for index, label in enumerate(self.labels):
            self.labels[index] = gzsl_to_zsl_label_indexes[label[0][0]]

    def fetch_batch(self, idx):
        im_name = self.file_names[idx][0][0][0].split('images/')[1]
        image_file = os.path.join(self.data_root, im_name)

        img_pil = Image.open(image_file).convert("RGB")
        label = self.labels[idx]
        label = label[0]
        label = torch.Tensor(label.astype(int))
        img_tensor = self.transform(img_pil)

        return (img_tensor, label)

    def __getitem__(self, idx):
        batch = self.fetch_batch(idx)

        return batch

from torch.utils.data import Dataset
from PIL import Image
import os
import torch

label_map = {
150 :  "001.Black_footed_Albatross", 0 :  "002.Laysan_Albatross",   138 :  "188.Pileated_Woodpecker",
1 :  "003.Sooty_Albatross", 151 :  "004.Groove_billed_Ani", 2 :  "005.Crested_Auklet",
152 :  "006.Least_Auklet", 3 :  "007.Parakeet_Auklet", 153 :  "008.Rhinoceros_Auklet", 154 :  "009.Brewer_Blackbird",
4 :  "010.Red_winged_Blackbird", 5 :  "011.Rusty_Blackbird", 6 :  "012.Yellow_headed_Blackbird", 7 :  "013.Bobolink",
155 :  "014.Indigo_Bunting", 8 :  "015.Lazuli_Bunting", 9 :  "016.Painted_Bunting",
10 :  "017.Cardinal", 11 :  "018.Spotted_Catbird", 12 :  "019.Gray_Catbird", 13 :  "020.Yellow_breasted_Chat",
14 :  "021.Eastern_Towhee", 15 :  "022.Chuck_will_Widow", 156 :  "023.Brandt_Cormorant", 16 :  "024.Red_faced_Cormorant",
17 :  "025.Pelagic_Cormorant", 18 :  "026.Bronzed_Cowbird", 19 :  "027.Shiny_Cowbird", 20 :  "028.Brown_Creeper",
157 :  "029.American_Crow", 21 :  "030.Fish_Crow", 158 :  "031.Black_billed_Cuckoo", 22 :  "032.Mangrove_Cuckoo",
159 :  "033.Yellow_billed_Cuckoo", 160 :  "034.Gray_crowned_Rosy_Finch", 161 :  "035.Purple_Finch",
162 :  "036.Northern_Flicker", 163 :  "037.Acadian_Flycatcher", 164 :  "038.Great_Crested_Flycatcher",
23 :  "039.Least_Flycatcher", 24 :  "040.Olive_sided_Flycatcher", 25 :  "041.Scissor_tailed_Flycatcher", 26 :  "042.Vermilion_Flycatcher",
165 :  "043.Yellow_bellied_Flycatcher", 27 :  "044.Frigatebird", 28 :  "045.Northern_Fulmar", 29 :  "046.Gadwall",
30 :  "047.American_Goldfinch", 31 :  "048.European_Goldfinch", 166 :  "049.Boat_tailed_Grackle", 32 :  "050.Eared_Grebe",
167 :  "051.Horned_Grebe", 33 :  "052.Pied_billed_Grebe", 168 :  "053.Western_Grebe", 34 :  "054.Blue_Grosbeak",
35 :  "055.Evening_Grosbeak", 36 :  "056.Pine_Grosbeak", 37 :  "057.Rose_breasted_Grosbeak", 38 :  "058.Pigeon_Guillemot",
39 :  "059.California_Gull", 40 :  "060.Glaucous_winged_Gull", 41 :  "061.Heermann_Gull", 42 :  "062.Herring_Gull",
43 :  "063.Ivory_Gull", 44 :  "064.Ring_billed_Gull", 45 :  "065.Slaty_backed_Gull", 169 :  "066.Western_Gull",
46 :  "067.Anna_Hummingbird", 47 :  "068.Ruby_throated_Hummingbird", 48 :  "069.Rufous_Hummingbird", 49 :  "070.Green_Violetear",
50 :  "071.Long_tailed_Jaeger",170 :  "072.Pomarine_Jaeger", 55 :  "077.Tropical_Kingbird", 56 :  "078.Gray_Kingbird",
51 :  "073.Blue_Jay", 52 :  "074.Florida_Jay", 53 :  "075.Green_Jay", 54 :  "076.Dark_eyed_Junco",
171 :  "079.Belted_Kingfisher", 57 :  "080.Green_Kingfisher", 58 :  "081.Pied_Kingfisher", 59 :  "082.Ringed_Kingfisher",
172 :  "083.White_breasted_Kingfisher", 173 :  "084.Red_legged_Kittiwake",60 :  "085.Horned_Lark",174 :  "086.Pacific_Loon",
61 :  "087.Mallard", 62 :  "088.Western_Meadowlark", 63 :  "089.Hooded_Merganser", 64 :  "090.Red_breasted_Merganser",
175 :  "091.Mockingbird", 65 :  "092.Nighthawk", 66 :  "093.Clark_Nutcracker", 67 :  "094.White_breasted_Nuthatch",
176 :  "095.Baltimore_Oriole", 177 :  "096.Hooded_Oriole", 68 :  "097.Orchard_Oriole", 178 :  "098.Scott_Oriole",
69 :  "099.Ovenbird", 70 :  "100.Brown_Pelican",179 :  "101.White_Pelican", 180 :  "102.Western_Wood_Pewee",
181 :  "103.Sayornis", 71 :  "104.American_Pipit", 72 :  "105.Whip_poor_Will", 73 :  "106.Horned_Puffin",
74 :  "107.Common_Raven", 75 :  "108.White_necked_Raven", 76 :  "109.American_Redstart", 77 :  "110.Geococcyx",
78 :  "111.Loggerhead_Shrike", 182 :  "112.Great_Grey_Shrike", 79 :  "113.Baird_Sparrow", 183 :  "114.Black_throated_Sparrow",
80 :  "115.Brewer_Sparrow", 81 :  "116.Chipping_Sparrow", 82 :  "117.Clay_colored_Sparrow", 83 :  "118.House_Sparrow",
184 :  "119.Field_Sparrow",84 :  "120.Fox_Sparrow",185 :  "121.Grasshopper_Sparrow",85 :  "122.Harris_Sparrow",
86 :  "123.Henslow_Sparrow", 87 :  "124.Le_Conte_Sparrow", 88 :  "125.Lincoln_Sparrow", 89 :  "126.Nelson_Sharp_tailed_Sparrow",
90 :  "127.Savannah_Sparrow", 91 :  "128.Seaside_Sparrow",92 :  "129.Song_Sparrow", 186 :  "130.Tree_Sparrow",
93 :  "131.Vesper_Sparrow", 94 :  "132.White_crowned_Sparrow", 95 :  "133.White_throated_Sparrow", 96 :  "134.Cape_Glossy_Starling",
187 :  "135.Bank_Swallow", 97 :  "136.Barn_Swallow", 98 :  "137.Cliff_Swallow", 188 :  "138.Tree_Swallow",
99 :  "139.Scarlet_Tanager",100 :  "140.Summer_Tanager", 101 :  "141.Artic_Tern", 102 :  "142.Black_Tern", 103 :  "143.Caspian_Tern",
104 :  "144.Common_Tern", 105 :  "145.Elegant_Tern", 106 :  "146.Forsters_Tern", 189 :  "147.Least_Tern", 107 :  "148.Green_tailed_Towhee",
108 :  "149.Brown_Thrasher", 109 :  "150.Sage_Thrasher", 110 :  "151.Black_capped_Vireo", 111 :  "152.Blue_headed_Vireo",
112 :  "153.Philadelphia_Vireo",113 :  "154.Red_eyed_Vireo",114 :  "155.Warbling_Vireo",190 :  "156.White_eyed_Vireo",
115 :  "157.Yellow_throated_Vireo",116 :  "158.Bay_breasted_Warbler",117 :  "159.Black_and_white_Warbler", 118 :  "160.Black_throated_Blue_Warbler",
119 :  "161.Blue_winged_Warbler", 120 :  "162.Canada_Warbler", 191 :  "163.Cape_May_Warbler", 121 :  "164.Cerulean_Warbler",
192 :  "165.Chestnut_sided_Warbler", 193 :  "166.Golden_winged_Warbler", 122 :  "167.Hooded_Warbler", 123 :  "168.Kentucky_Warbler",
124 :  "169.Magnolia_Warbler", 125 :  "170.Mourning_Warbler", 126 :  "171.Myrtle_Warbler", 127 :  "172.Nashville_Warbler",
128 :  "173.Orange_crowned_Warbler", 129 :  "174.Palm_Warbler", 130 :  "175.Pine_Warbler", 131 :  "176.Prairie_Warbler",
132 :  "177.Prothonotary_Warbler", 133 :  "178.Swainson_Warbler",198 :  "187.American_Three_toed_Woodpecker",
134 :  "179.Tennessee_Warbler", 194 :  "180.Wilson_Warbler",135 :  "181.Worm_eating_Warbler",136 :  "182.Yellow_Warbler",
195 :  "183.Northern_Waterthrush", 137 :  "184.Louisiana_Waterthrush", 196 :  "185.Bohemian_Waxwing", 197 :  "186.Cedar_Waxwing",
139 :  "189.Red_bellied_Woodpecker",140 :  "190.Red_cockaded_Woodpecker", 141 :  "191.Red_headed_Woodpecker", 142 :  "192.Downy_Woodpecker",
143 :  "193.Bewick_Wren",144 :  "194.Cactus_Wren",145 :  "195.Carolina_Wren", 146 :  "196.House_Wren",
199 :  "197.Marsh_Wren", 147 :  "198.Rock_Wren", 148 :  "199.Winter_Wren", 149 :  "200.Common_Yellowthroat"
}

unseen_indices = [6,18,20,28,33,35,49,55,61,67,68,71,78,79,86,87,90,94
,97,99,103,107,115,119,121,123,124,128,138,140,141,149,151,156,158,159
,165,166,170,173,175,178,181,184,186,188,190,191,192,194]

seen_indices = [x for x in range(len(label_map)) if x not in unseen_indices]


zsl_to_gzsl_label_indexes= {0: 6, 1: 18, 2: 20, 3: 28, 4: 33, 5: 35, 6: 49, 7: 55, 8: 61, 9: 67, 10: 68, 11: 71, 12: 78,
13: 79, 14: 86, 15: 87, 16: 90, 17: 94, 18: 97, 19: 99, 20: 103, 21: 107, 22: 115, 23: 119, 24: 121, 25: 123, 26: 124, 27:
128, 28: 138, 29: 140, 30: 141, 31: 149, 32: 151, 33: 156, 34: 158, 35: 159, 36: 165, 37: 166, 38: 170, 39: 173, 40: 175,
41: 178, 42: 181, 43: 184, 44: 186, 45: 188, 46: 190, 47: 191, 48: 192, 49: 194}

gzsl_to_zsl_label_indexes = {6: 0, 18: 1, 20: 2, 28: 3, 33: 4, 35: 5, 49: 6, 55: 7, 61: 8, 67: 9, 68: 10, 71: 11, 78: 12,
79: 13, 86: 14, 87: 15, 90: 16, 94: 17, 97: 18, 99: 19, 103: 20, 107: 21, 115: 22, 119: 23, 121: 24, 123: 25, 124: 26,
128: 27, 138: 28, 140: 29, 141: 30, 149: 31, 151: 32, 156: 33, 158: 34, 159: 35, 165: 36, 166: 37, 170: 38, 173: 39, 175: 40,
178: 41, 181: 42, 184: 43, 186: 44, 188: 45, 190: 46, 191: 47, 192: 48, 194: 49}



class CUBDataset(Dataset):
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
        label = torch.Tensor(label)
        img_tensor = self.transform(img_pil)

        return (img_tensor, label)

    def __getitem__(self, idx):
        batch = self.fetch_batch(idx)

        return batch

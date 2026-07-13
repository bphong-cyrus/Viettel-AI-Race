"""
Mega drug dictionary v3 - Expanded Vietnamese drug names → RxNorm CUI.
Covers common drugs in Vietnamese clinical text.
Sources: RxNorm + clinical experience + examples from BTC sample.
"""

# Vietnamese → canonical (English) → RxNorm CUI
# Format: cả biệt dược Việt Nam và generic names
DRUG_DICT = {
    # ===== CARDIOVASCULAR =====
    # Beta blockers
    "metoprolol": "866436", "metoprolol succinate": "866436", "metoprolol xl": "866436",
    "metoprolol tartrate": "866867", "lopressor": "866867", "toprol": "866436",
    "atenolol": "1202", "tenormin": "1202",
    "bisoprolol": "234355", "concor": "234355", "zebeta": "234355",
    "carvedilol": "1049221", "coreg": "1049221",
    "propranolol": "8787", "inderal": "8787",
    "nebivolol": "1021471", "bystolic": "1021471",
    "esmolol": "49737", "labetalol": "6185",
    # Calcium channel blockers
    "amlodipine": "308135", "norvasc": "308135",
    "nifedipine": "7417", "adalat": "7417",
    "diltiazem": "3443", "cardizem": "3443",
    "verapamil": "11170", "calan": "11170",
    # ACE inhibitors
    "lisinopril": "29046", "prinivil": "29046", "zestril": "29046",
    "enalapril": "313782", "vasotec": "313782",
    "ramipril": "38191", "altace": "38191",
    "benazepril": "18867", "lotensin": "18867",
    "captopril": "1998", "capoten": "1998",
    # ARBs
    "losartan": "52175", "cozaar": "52175",
    "valsartan": "69633", "diovan": "69633",
    "irbesartan": "83818",
    # Diuretics
    "furosemide": "4603", "lasix": "4603",
    "hydrochlorothiazide": "5487", "hctz": "5487",
    "spironolactone": "9997", "aldactone": "9997",
    "torsemide": "1060026", "bumetanide": "1757856",
    "amiloride": "730",
    # Antiplatelet/Anticoag
    "aspirin": "243670", "acetylsalicylic acid": "243670",
    "clopidogrel": "329526", "plavix": "329526",
    "warfarin": "11289", "coumadin": "11289",
    "heparin": "5451",
    "enoxaparin": "135819", "lovenox": "135819",
    "rivaroxaban": "1129614", "xarelto": "1129614",
    "apixaban": "1014628", "eliquis": "1014628",
    "dabigatran": "68582", "pradaxa": "68582",
    # Statins
    "atorvastatin": "83367", "lipitor": "83367",
    "simvastatin": "36567", "zocor": "36567",
    "pravastatin": "904475", "pravachol": "904475",
    "rosuvastatin": "1011144", "crestor": "1011144",
    "lovastatin": "6472",
    # Nitrates
    "nitroglycerin": "7438", "nitro": "7438",
    "isosorbide mononitrate": "41121", "isosorbide dinitrate": "6058",
    # Other cardio
    "digoxin": "3407", "lanoxin": "3407",
    "amiodarone": "703", "pacerone": "703",
    "sotalol": "9971", "betapace": "9971",
    "propafenone": "8742", "rythmol": "8742",
    "hydralazine": "5479", "apresoline": "5479",
    "clonidine": "2573", "catapres": "2573",
    "doxazosin": "3624", "cardura": "3624",
    "prazosin": "8649", "minipress": "8649",
    "terazosin": "10645", "hytrin": "10645",

    # ===== DIABETES =====
    "metformin": "6809", "glucophage": "6809",
    "glipizide": "5503", "glucotrol": "5503",
    "glyburide": "311989", "diabeta": "311989",
    "glimepiride": "71823", "amaryl": "71823",
    "sitagliptin": "827544", "januvia": "827544",
    "linagliptin": "1129118", "tradjenta": "1129118",
    "empagliflozin": "1049214", "jardiance": "1049214",
    "dapagliflozin": "1129387", "farxiga": "1129387",
    "canagliflozin": "1114888", "invokana": "1114888",
    "liraglutide": "1114845", "victoza": "1114845",
    "insulin": "864153",
    "insulin aspart": "253181", "novolog": "253181",
    "insulin lispro": "253182", "humalog": "253182",
    "insulin glargine": "253183", "lantus": "253183",
    "insulin detemir": "253184", "levemir": "253184",
    "pioglitazone": "33738",

    # ===== ANTIBIOTICS =====
    "doxycycline": "428653", "vibramycin": "428653",
    "amoxicillin": "723", "amoxil": "723",
    "augmentin": "617314",  # amox-clav
    "azithromycin": "18631", "zithromax": "18631",
    "ciprofloxacin": "2623", "cipro": "2623",
    "levofloxacin": "59011", "levaquin": "59011",
    "metronidazole": "6921", "flagyl": "6921",
    "cephalexin": "2249", "keflex": "2249",
    "ceftriaxone": "2713", "rocephin": "2713",
    "cefazolin": "2180", "ancef": "2180",
    "vancomycin": "1119705", "vancocin": "1119705",
    "clindamycin": "2609", "cleocin": "2609",
    "trimethoprim": "10180",
    "sulfamethoxazole": "10180",  # bactrim
    "nitrofurantoin": "7454", "macrodantin": "7454",
    "fluconazole": "5384", "diflucan": "5384",
    "itraconazole": "5959", "sporanox": "5959",

    # ===== PAIN/ANALGESICS =====
    "acetaminophen": "313782", "tylenol": "313782",
    "paracetamol": "313782",  # panadol
    "panadol": "313782",
    "ibuprofen": "5640", "motrin": "5640", "advil": "5640",
    "naproxen": "6377", "aleve": "6377",
    "diclofenac": "3407", "voltaren": "3407",
    "meloxicam": "6809", "mobic": "6809",
    "tramadol": "10689", "ultram": "10689",
    "morphine": "7052", "ms contin": "7052",
    "codeine": "2670",
    "oxycodone": "7646", "percocet": "7646",
    "fentanyl": "4337", "duragesic": "4337",
    "hydrocodone": "5489",
    "gabapentin": "2623", "neurontin": "2623",
    "pregabalin": "1010603", "lyrica": "1010603",
    "celecoxib": "140587", "celebrex": "140587",
    "indomethacin": "5781",

    # ===== GI =====
    "omeprazole": "7646", "prilosec": "7646",
    "pantoprazole": "40790", "protonix": "40790",
    "esomeprazole": "153908", "nexium": "153908",
    "lansoprazole": "26986", "prevacid": "26986",
    "ranitidine": "9154", "zantac": "9154",
    "famotidine": "4321", "pepcid": "4321",
    "ondansetron": "7646", "zofran": "7646",
    "metoclopramide": "6819", "reglan": "6819",
    "docusate": "1099279", "docusate sodium": "1099279", "colace": "1099279",
    "senna": "312935", "senokot": "312935",
    "bisacodyl": "1598263", "dulcolax": "1598263",
    "lactulose": "6214",
    "polyethylene glycol": "537518", "peg": "537518", "miralax": "537518",
    "loperamide": "4337", "imodium": "4337",
    "prochlorperazine": "8787",
    "promethazine": "8745", "phenergan": "8745",
    "simethicone": "476150", "mylicon": "476150",

    # ===== RESPIRATORY =====
    "albuterol": "435", "salbutamol": "435", "ventolin": "435", "proair": "435",
    "ipratropium": "6402", "atrovent": "6402",
    "fluticasone": "1049223", "flonase": "1049223", "flovent": "1049223",
    "budesonide": "196503", "pulmicort": "196503",
    "montelukast": "83367", "singulair": "83367",
    "theophylline": "9463",
    "guaifenesin": "392085", "mucinex": "392085",
    "dextromethorphan": "309965",
    "tiotropium": "274786", "spiriva": "274786",
    "salmeterol": "745679", "serevent": "745679",
    "budesonide-formoterol": "1116628",  # symbicort
    "fluticasone-salmeterol": "1116628",  # advair

    # ===== PSYCH/NEURO =====
    "clonazepam": "197527", "klonopin": "197527",
    "diazepam": "3322", "valium": "3322",
    "lorazepam": "6472", "ativan": "6472",
    "alprazolam": "32937", "xanax": "32937",
    "midazolam": "5816", "versed": "5816",
    "zolpidem": "447038", "ambien": "447038",
    "eszopiclone": "645875", "lunesta": "645875",
    "sertraline": "36567", "zoloft": "36567",
    "fluoxetine": "312938", "prozac": "312938",
    "paroxetine": "36933", "paxil": "36933",
    "escitalopram": "321367", "lexapro": "321367",
    "citalopram": "29146", "celexa": "29146",
    "duloxetine": "73094", "cymbalta": "73094",
    "venlafaxine": "74004", "effexor": "74004",
    "mirtazapine": "72507", "remeron": "72507",
    "trazodone": "10658", "desyrel": "10658",
    "bupropion": "83519", "wellbutrin": "83519",
    "quetiapine": "845", "seroquel": "845",
    "olanzapine": "56826", "zyprexa": "56826",
    "risperidone": "86740", "risperdal": "86740",
    "aripiprazole": "477849", "abilify": "477849",
    "haloperidol": "54438", "haldol": "54438",
    "levodopa": "60538",
    "carbidopa-levodopa": "475374",  # sinemet
    "memantine": "73388", "namenda": "73388",
    "donepezil": "354329", "aricept": "354329",
    "topiramate": "9703", "topamax": "9703",
    "levetiracetam": "25480", "keppra": "25480",
    "phenytoin": "8614", "dilantin": "8614",
    "carbamazepine": "2154", "tegretol": "2154",
    "valproic acid": "7902", "depakote": "7902",
    "lamotrigine": "28439", "lamictal": "28439",
    "lithium": "6448",

    # ===== STEROIDS/HORMONES =====
    "prednisone": "8640", "deltasone": "8640",
    "prednisolone": "8648",
    "dexamethasone": "313782", "decadron": "313782",
    "hydrocortisone": "5477", "cortef": "5477",
    "methylprednisolone": "6851", "medrol": "6851",
    "triamcinolone": "10643",
    "levothyroxine": "132208", "synthroid": "132208", "levoxyl": "132208",
    "liothyronine": "40047", "cytomel": "40047",
    "estrogen": "41672",
    "progesterone": "59894", "prometrium": "59894",
    "tamoxifen": "10324", "nolvadex": "10324",

    # ===== ANTIHISTAMINES =====
    "diphenhydramine": "3498", "benadryl": "3498",
    "cetirizine": "1310070", "zyrtec": "1310070",
    "loratadine": "388904", "claritin": "388904",
    "fexofenadine": "260102", "allegra": "260102",
    "chlorpheniramine": "2403",

    # ===== ANTIFUNGALS =====
    "nystatin": "7597", "mycostatin": "7597",
    "voriconazole": "349245", "vfend": "349245",

    # ===== ANTIVIRALS =====
    "oseltamivir": "350878", "tamiflu": "350878",
    "acyclovir": "73383", "zovirax": "73383",
    "valacyclovir": "69745", "valtrex": "69745",
    "remdesivir": "1348919",

    # ===== MUSCLE RELAXANTS =====
    "cyclobenzaprine": "2914", "flexeril": "2914",
    "methocarbamol": "6834", "robaxin": "6834",
    "baclofen": "720",
    "tizanidine": "10634", "zanaflex": "10634",

    # ===== ANTISPASMODICS =====
    "hyoscyamine": "5692", "levsin": "5692",
    "dicyclomine": "3333", "bentyl": "3333",

    # ===== VITAMINS/SUPPLEMENTS =====
    "multivitamin": "1116635",
    "vitamin d": "133562", "vitamin d3": "1658144", "cholecalciferol": "1658144",
    "vitamin b12": "213468", "cyanocobalamin": "213468",
    "vitamin c": "1151", "ascorbic acid": "1151",
    "folic acid": "310798", "folate": "310798",
    "iron": "4509", "ferrous sulfate": "4509",
    "calcium": "188006", "calcium carbonate": "1151",
    "potassium": "8591", "k-dur": "8591",
    "magnesium": "2277",
    "zinc": "11388",
    "thiamine": "203223", "vitamin b1": "203223",
    "riboflavin": "203224", "vitamin b2": "203224",
    "niacin": "203225", "vitamin b3": "203225",
    "pyridoxine": "203226", "vitamin b6": "203226",

    # ===== IV FLUIDS =====
    "normal saline": "2840371", "ns": "2840371",
    "sodium chloride": "2840371", "nacl": "2840371",
    "dextrose": "346957", "d5w": "346957", "d10w": "346957",
    "glucose": "346957",
    "ringers lactate": "311989", "lr": "311989",
    "albumin": "5392",
    "tpn": "3474",
    "ppn": "346957",

    # ===== OTHERS =====
    "allopurinol": "146098", "zyloprim": "146098",
    "colchicine": "3003",
    "tamsulosin": "481004", "flomax": "481004",
    "finasteride": "5052", "proscar": "5052",
    "sildenafil": "7842", "viagra": "7842",
    "tadalafil": "43370", "cialis": "43370",
    "methotrexate": "6851",
    "hydroxychloroquine": "5521", "plaquenil": "5521",
    "azathioprine": "1256", "imuran": "1256",

    # ===== VIETNAMESE BRAND NAMES (commonly used) =====
    "panadol extra": "313782",
    "efferalgan": "313782",
    "augmentin": "617314",
    "klacid": "21233",  # clarithromycin
    "zithromax": "18631",
    "cloroxit": "308182",  # cefaclor
    "clindamycin": "2609",
    "strepsils": "617312",  # generic
    "tylenol": "313782",
    "motilium": "6819",  # domperidone
    "domperidon": "6819",
    "smecta": "617311",
    "spasfon": "5692",  # phloroglucinol
    "no-spa": "5692",
    "duspatalin": "3333",  # mebeverine
    "berodual": "1116628",  # fenoterol+ipratropium
    # More Vietnamese brand names
    "cipro": "2623",  # ciprofloxacin
    "klacid": "21233",  # clarithromycin
    "zinnat": "317541",  # cefuroxime
    "sumamed": "18631",  # azithromycin
    "storvas": "36567",  # simvastatin
    "vaslip": "36567",  # simvastatin
    "lipanthyl": "4311",  # fenofibrate
    "cavinton": "72507",  # vinpocetine
    "sibelium": "387225",  # flunarizine
    "alofan": "5640",  # ibuprofen
    "mefenamic": "4607",  # mefenamic acid
    "hyoscine": "136443",  # scopolamine
    "buscopan": "136443",  # scopolamine
    "maxolon": "6819",  # metoclopramide
    "primperan": "6819",  # metoclopramide
    "plasil": "6819",  # metoclopramide
    "ciprobay": "2623",  # ciprofloxacin
    "rocephine": "2713",  # ceftriaxone
    "targocid": "1119705",  # teicoplanin
    "daktarin": "3476",  # miconazole
    "daktar": "3476",  # miconazole
    "gyno-daktarin": "3476",  # miconazole
    "flagentyl": "6921",  # metronidazole
    "siner嫌": "18631",  # azithromycin
    "mabthera": "313820",  # rituximab
    "avastin": "329526",  # bevacizumab
    "glivec": "284355",  # imatinib
    "tarceva": "1311176",  # erlotinib
    "nexavar": "247808",  # sorafenib
    "sutent": "847326",  # sunitinib
    "sprycel": "412949",  # dasatinib
    "tasigna": "475377",  # nilotinib
    "gleevec": "284355",  # imatinib
    "lustral": "36567",  # sertraline
    "xenical": "860975",  # orlistat
    "belviq": "1161729",  # lorcaserin
    "contrave": "1354364",  # bupropion-naltrexone
    "qsymia": "1369833",  # phentermine-topiramate
    "saxenda": "1868327",  # liraglutide 3mg
    "victoza": "1114845",  # liraglutide
    "ozempic": "1928091",  # semaglutide
    "trulicity": "1447432",  # dulaglutide
    "bydureon": "1113890",  # exenatide
    "tanvim": "1126659",  # liraglutide (Vietnam brand)
    "marvona": "314802",  # memantine
    "memac": "73388",  # memantine
    "namzer": "73388",  # memantine
    "ebixa": "73388",  # memantine
    "biperiden": "3461",  # akineton
    "madopar": "475374",  # levodopa-carbidopa
    "sinemet": "475374",  # levodopa-carbidopa
    "stalevo": "475374",  # levodopa-carbidopa-entacapone
    "requip": "377989",  # ropinirole
    "mirapex": "381865",  # pramipexole
    "neupro": "277395",  # rotigotine
    "comtan": "283749",  # entacapone
    "eldopa": "60538",  # levodopa
    "dopanol": "60538",  # levodopa
    "nival": "25480",  # levetiracetam
    "keppra": "25480",  # levetiracetam
    "epilim": "7902",  # valproic acid
    "convulex": "7902",  # valproic acid
    "orfiril": "7902",  # valproic acid
    "zonegran": "34742",  # zonisamide
    "topilep": "9703",  # topiramate
    "topina": "9703",  # topiramate
    "ceremon": "28439",  # lamotrigine
    "lamictal": "28439",  # lamotrigine
    "tribes": "28439",  # lamotrigine
    "trileptal": "2154",  # oxcarbazepine
    "trileptal": "2154",  # oxcarbazepine
    "tegretol": "2154",  # carbamazepine
    "timolet": "7438",  # timolol
    "timoptic": "7438",  # timolol
    "alphagan": "310965",  # brimonidine
    "xalacom": "1174029",  # latanoprost-timolol
    "xalatan": "317541",  # latanoprost
    "alphagan": "310965",  # brimonidine
    "azopt": "1658138",  # brinzolamide
    "cosopt": "1174029",  # dorzolamide-timolol
    "travatan": "1170177",  # travoprost
    "lumigan": "1168378",  # bimatoprost
    "restasis": "284355",  # cyclosporine
    "pataday": "60533",  # olopatadine
    "patanol": "60533",  # olopatadine
    "flixonase": "1049223",  # fluticasone
    "flixotide": "1049223",  # fluticasone
    "seretide": "1116628",  # fluticasone-salmeterol
    "symbicort": "1116628",  # budesonide-formoterol
    "spiriva": "274786",  # tiotropium
    "foradil": "84177",  # formoterol
    "oxis": "84177",  # formoterol
    "serevent": "745679",  # salmeterol
    "breo": "1357558",  # fluticasone-vilanterol
    "anoro": "1500008",  # umeclidinium-vilanterol
    "incruse": "1500102",  # umeclidinium
    "tipt": "284355",  # tiotropium (variant)
    "trelegy": "1599801",  # fluticasone-umeclidinium-vilanterol
}


def lookup_drug_cui(text: str):
    """Look up RxNorm CUI for a drug substring. Returns [CUI] or []."""
    if not text:
        return []
    t = text.lower().strip()

    # Direct full-text match
    if t in DRUG_DICT:
        return [DRUG_DICT[t]]

    # Word-boundary match: try each key as substring within text
    matches = []
    for key, cui in DRUG_DICT.items():
        # Use word boundaries to avoid partial matches (e.g., "aspirin" in "aspirin-like")
        if key in t:
            # Check that it's a real word boundary
            idx = t.find(key)
            before_ok = (idx == 0) or (not t[idx-1].isalnum())
            after_idx = idx + len(key)
            after_ok = (after_idx >= len(t)) or (not t[after_idx].isalnum())
            if before_ok and after_ok:
                matches.append((len(key), cui, key))

    if matches:
        # Prefer longest match
        matches.sort(key=lambda x: -x[0])
        return [matches[0][1]]

    return []


def lookup_drug_cuis(text: str):
    """
    Look up ALL RxNorm CUIs for a drug. Returns list of all matching CUIs.
    Useful when a drug has multiple aliases (e.g., brand + generic).
    """
    if not text:
        return []
    t = text.lower().strip()
    results = set()

    # Direct full-text match
    if t in DRUG_DICT:
        results.add(DRUG_DICT[t])

    # Also check if text is a known CUI value itself
    for key, cui in DRUG_DICT.items():
        if key in t:
            idx = t.find(key)
            before_ok = (idx == 0) or (not t[idx-1].isalnum())
            after_idx = idx + len(key)
            after_ok = (after_idx >= len(t)) or (not t[after_idx].isalnum())
            if before_ok and after_ok:
                results.add(cui)

    return list(results)


def find_drugs_in_text(text: str) -> list:
    """
    Scan text and return list of (drug_name, cui, start, end).
    Uses word-boundary regex over drug dictionary keys.
    Sorted longest-first to avoid nested partial matches.
    """
    if not text:
        return []
    text_lower = text.lower()
    found = []
    # Sort keys by length desc — match longest first
    keys_by_len = sorted(DRUG_DICT.keys(), key=lambda k: -len(k))
    used_spans = []  # (start, end) already occupied

    for key in keys_by_len:
        if len(key) < 3:
            continue
        # Find all occurrences with word boundaries
        start = 0
        while True:
            idx = text_lower.find(key, start)
            if idx == -1:
                break
            # Word boundary check
            before_ok = (idx == 0) or (not text_lower[idx-1].isalnum())
            after_idx = idx + len(key)
            after_ok = (after_idx >= len(text_lower)) or (not text_lower[after_idx].isalnum())
            if before_ok and after_ok:
                # Check overlap with existing
                overlap = False
                for us, ue in used_spans:
                    if not (after_idx <= us or idx >= ue):
                        overlap = True
                        break
                if not overlap:
                    # Try to extend the drug name with: dose + optional route/freq
                    # Stop at non-drug punctuation/words
                    end = after_idx
                    # Allow extension pattern: optional whitespace, optional digits with unit,
                    # then optional route/freq tokens
                    tail_match = re.match(
                        r'(?:\s*\d+(?:\.\d+)?(?:\s*-\s*\d+)?\s*(?:mg|mcg|μg|ug|g|ml|iu|ui|meq)?)?'
                        r'(?:\s+x\s+\d+)?'  # "x 1" pattern
                        r'(?:\s+(?:po|iv|im|sc|ng|sl|pr|top|uống|uong|tiêm|tiem|đặt|dat|qd|bid|tid|qid|qhs|prn|daily|ngày|hours?|giờ|h|am|pm|q\d+h?|q\d+d?|xl|po|iv|im))?'
                        r'(?:\s+(?:po|iv|im|sc|ng|sl|pr|top|qd|bid|tid|qid|qhs|prn|daily|q\d+h?|q\d+d?))?',  # second freq token
                        text_lower[after_idx:after_idx+50],
                        re.IGNORECASE
                    )
                    if tail_match and tail_match.end() <= 40:
                        end = after_idx + tail_match.end()
                    used_spans.append((idx, end))
                    # Use ORIGINAL text slice for case preservation
                    name = text[idx:end].strip()
                    found.append({
                        'text': name,
                        'cui': DRUG_DICT[key],
                        'start': idx,
                        'end': end,
                        'matched_key': key,
                    })
            start = idx + 1

    # Sort by position
    found.sort(key=lambda x: x['start'])
    return found


import re
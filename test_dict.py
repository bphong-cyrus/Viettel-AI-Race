"""
Test name and result dictionary.
Auto-built from v11r3 (38.99 baseline).
"""
import re

# Common TÊN_XÉT_NGHIỆM (Test names) - frequency >= 2 from v11r3
TEST_NAMES = [
    # High frequency
    "nội soi", "siêu âm", "x-quang ngực", "men gan", "chụp cắt lớp vi tính",
    "sinh thiết", "ast", "alt", "ct", "chụp ct", "chụp ct sọ não",
    "tổng phân tích nước tiểu", "điện tâm đồ", "ercp", "chọc dò dịch não tủy",
    "nội soi dạ dày", "troponin", "siêu âm tim", "đông máu", "cấy máu",
    "siêu âm bụng", "nội soi mật tụy ngược dòng (ercp)", "x-quang",
    "điện tim", "ecg", "hba1c", "glucose", "creatinin", "creatinine",
    "siêu âm doppler", "chụp ct bụng", "ct ngực", "ct sọ não",
    "chức năng gan", "albumin", "alp", "bilirubin toàn phần", "bnp",
    "cholangiogram", "cl", "cl-", "crp", "cộng hưởng từ mật tụy",
    "chụp cộng hưởng từ mật tụy", "chụp hida", "chụp động mạch vành",
    "các băng nhóm oligoclonal", "cúm", "cấy nước tiểu", "got", "gpt",
    "guaiac", "hco3", "hemoglobin", "huyết khối", "inr", "k", "k+",
    "kali", "kali (k)", "khí máu", "lactate", "lấy mẫu bằng bàn chải",
    "mri", "mô bệnh học", "na", "na+", "nghiệm pháp gắng sức",
    "nitrite", "nt - probnp", "pco2", "ph", "phân tích nước tiểu",
    "po2", "protein máu", "protein niệu 24h", "pt", "số lượng bạch cầu",
    "tq", "triglycerid", "tổng phân tích tế bào máu ngoại vi",
    "tỷ lệ prothrombin", "ure", "ure (bun)", "xquang ngực thẳng",
    "xét nghiệm gắng sức", "xét nghiệm nước tiểu", "xét nghiệm phân",
    "điện giải đồ", "đo sức bền cơ tim bằng đồng vị phóng xạ",
    "đo thính lực", "đường huyết", "bạch cầu",
    "ef bp", "hiv", "hcv", "hbsag", "anti-hbs",
    "chụp mri", "xét nghiệm máu", "nước tiểu", "phân tích máu",
    "công thức máu", "định nhóm máu", "đông máu cơ bản", "đường huyết",
    "ure máu", "creatinine máu", "điện giải", "ion đồ",
    "đo hoạt độ alanin aminotransferase", "đo hoạt độ aspartate aminotransferase",
    "đo nồng độ bilirubin toàn phần", "đo nồng độ protein",
    "đo nồng độ albumin", "đo tỷ lệ prothrombin",
    "thời gian thromboplastin", "thời gian prothrombin",
    "tế bào học", "mô bệnh học", "hóa mô miễn dịch",
    "nhuộm soi", "nhuộm gram", "soi tươi", "soi phân",
    "phản ứng mantoux", "phản ứng tuberculin", "test da",
    "pap", "cell block", "tế bào cổ tử cung",
    "đo thị lực", "đo nhãn áp", "soi đáy mắt", "chụp võng mạc",
    "đo chức năng hô hấp", "đo phế dung", "test hồi phục phế quản",
    "nội soi phế quản", "nội soi đại tràng", "nội soi bàng quang",
    "nội soi tai mũi họng", "nội soi thanh quản",
    "chụp x-quang", "chụp cắt lớp", "chụp cộng hưởng từ",
    "chụp mạch", "chụp mạch vành", "chụp động mạch",
    "siêu âm doppler tim", "siêu âm thai", "siêu âm 4d",
    "siêu âm tuyến giáp", "siêu âm vú", "siêu âm gan",
    "siêu âm thận", "siêu âm lách", "siêu âm tử cung phần phụ",
    "xét nghiệm kháng đông", "xét nghiệm đông máu", "xét nghiệm tế bào",
    "xét nghiệm dị ứng", "xét nghiệm nội tiết", "xét nghiệm di truyền",
    "định lượng kháng thể", "định nhóm máu hệ abo", "định nhóm máu hệ rh",
    "đo đường huyết", "đo huyết áp", "đo thân nhiệt",
    "đo sp02", "đo spo2", "monitor", "ecg monitor",
    "khám", "khám lâm sàng", "khám chuyên khoa",
]

# Common KẾT_QUẢ_XÉT_NGHIỆM (Test results)
TEST_RESULTS = [
    "bình thường", "bất thường", "tăng", "giảm", "âm tính", "dương tính",
    "+", "-", "(+)", "(-)", "thấp", "cao",
    "không có gì đáng chú ý", "không thấy gì cả", "không phát hiện",
    "âm tính với", "dương tính với", "dương tính yếu",
    "âm tính giả", "dương tính giả",
    "vếtvết protein niệu", "protein niệu", "huyết sắc tố",
    "không đặc hiệu", "không rõ", "không xác định",
]

# Sort by length (longest first) for longest-match-first search
TEST_NAMES_SORTED = sorted(set(TEST_NAMES), key=lambda x: -len(x))
TEST_RESULTS_SORTED = sorted(set(TEST_RESULTS), key=lambda x: -len(x))


# Section indicators for TÊN_XÉT_NGHIỆM
TEST_SECTION_PATTERNS = [
    r"xét\s+nghiệm",
    r"cận\s+lâm\s+sàng",
    r"chẩn\s+đoán\s+hình\s+ảnh",
    r"kết\s+quả\s+xét\s+nghiệm",
    r"kết\s+quả\s+cận\s+lâm\s+sàng",
    r"kết\s+quả\s+chẩn\s+đoán\s+hình\s+ảnh",
    r"xét\s+nghiệm\s+máu",
    r"xét\s+nghiệm\s+nước\s+tiểu",
    r"chụp\s+(?:ct|mri|x[-\s]?quang|siêu\s+âm)",
    r"siêu\s+âm",
    r"nội\s+soi",
    r"sinh\s+thiết",
]

TEST_SECTION_RE = re.compile("|".join(TEST_SECTION_PATTERNS), re.IGNORECASE)

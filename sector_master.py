# 東証33業種分類システム
# 証券コードの範囲から業種を自動判定

def get_sector_from_code(code):
    """
    証券コードから東証33業種を判定
    
    Args:
        code: 証券コード（文字列または数値）
    
    Returns:
        業種名（文字列）
    """
    try:
        code_str = str(code).split('.')[0].strip()
        code_num = int(code_str)
    except:
        return "その他"
    
    # 東証コード範囲による業種分類
    if 1300 <= code_num <= 1499:
        return "水産・農林業" if code_num < 1400 else "鉱業"
    
    elif 1500 <= code_num <= 1999:
        return "建設業"
    
    elif 2000 <= code_num <= 2999:
        return "食料品"
    
    elif 3000 <= code_num <= 3499:
        return "繊維製品"
    
    elif 3500 <= code_num <= 3799:
        return "パルプ・紙"
    
    elif 3800 <= code_num <= 4499:
        return "化学"
    
    elif 4500 <= code_num <= 4699:
        return "医薬品"
    
    elif 4700 <= code_num <= 4899:
        return "情報・通信業"
    
    elif 5000 <= code_num <= 5099:
        return "石油・石炭製品"
    
    elif 5100 <= code_num <= 5199:
        return "ゴム製品"
    
    elif 5200 <= code_num <= 5399:
        return "ガラス・土石製品"
    
    elif 5400 <= code_num <= 5599:
        return "鉄鋼"
    
    elif 5600 <= code_num <= 5799:
        return "非鉄金属"
    
    elif 5800 <= code_num <= 6099:
        return "金属製品"
    
    elif 6100 <= code_num <= 6399:
        return "機械"
    
    elif 6400 <= code_num <= 6999:
        return "電気機器"
    
    elif 7000 <= code_num <= 7499:
        return "輸送用機器"
    
    elif 7500 <= code_num <= 7899:
        return "精密機器"
    
    elif 7900 <= code_num <= 7999:
        return "その他製品"
    
    elif 8000 <= code_num <= 8299:
        return "卸売業"
    
    elif 8300 <= code_num <= 8499:
        return "銀行業"
    
    elif 8500 <= code_num <= 8599:
        return "その他金融業"
    
    elif 8600 <= code_num <= 8699:
        return "証券業・商品先物取引業"
    
    elif 8700 <= code_num <= 8799:
        return "保険業"
    
    elif 8800 <= code_num <= 8999:
        return "不動産業"
    
    elif 9000 <= code_num <= 9099:
        return "陸運業"
    
    elif 9100 <= code_num <= 9199:
        return "海運業"
    
    elif 9200 <= code_num <= 9299:
        return "空運業"
    
    elif 9300 <= code_num <= 9399:
        return "倉庫・運輸関連業"
    
    elif 9400 <= code_num <= 9499:
        return "情報・通信業"
    
    elif 9500 <= code_num <= 9599:
        return "電気・ガス業"
    
    elif 9600 <= code_num <= 9799:
        return "サービス業"
    
    elif 9800 <= code_num <= 9999:
        return "小売業"
    
    else:
        return "その他"


# 東証33業種の一覧
TSE_33_SECTORS = [
    "水産・農林業",
    "鉱業",
    "建設業",
    "食料品",
    "繊維製品",
    "パルプ・紙",
    "化学",
    "医薬品",
    "石油・石炭製品",
    "ゴム製品",
    "ガラス・土石製品",
    "鉄鋼",
    "非鉄金属",
    "金属製品",
    "機械",
    "電気機器",
    "輸送用機器",
    "精密機器",
    "その他製品",
    "電気・ガス業",
    "陸運業",
    "海運業",
    "空運業",
    "倉庫・運輸関連業",
    "情報・通信業",
    "卸売業",
    "小売業",
    "銀行業",
    "証券業・商品先物取引業",
    "保険業",
    "その他金融業",
    "不動産業",
    "サービス業",
]


def test_sector_mapping():
    """テスト用：主要銘柄の業種判定を確認"""
    test_codes = {
        "5334": "日本特殊陶業",
        "8919": "カチタス",
        "7203": "トヨタ自動車",
        "6758": "ソニーグループ",
        "9984": "ソフトバンクグループ",
        "4502": "武田薬品工業",
        "8306": "三菱UFJ",
        "3382": "セブン&アイ",
        "9983": "ファーストリテイリング",
        "4063": "信越化学工業",
    }
    
    print("=" * 60)
    print("業種判定テスト")
    print("=" * 60)
    
    for code, name in test_codes.items():
        sector = get_sector_from_code(code)
        print(f"{code}: {name:25s} → {sector}")
    
    print("=" * 60)


if __name__ == "__main__":
    test_sector_mapping()

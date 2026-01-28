# 🏴‍☠️ お宝発掘ツール - 新規デプロイガイド

## 📦 必要なファイル

新しいアプリとして公開するために、以下の4ファイルをGitHubにアップロードしてください。

```
treasure-hunter/  ← 新しいリポジトリ名（例）
├── app.py
├── sector_master.py
├── requirements.txt
└── README.md
```

---

## 🚀 デプロイ手順

### ステップ1: 新しいGitHubリポジトリを作成

1. https://github.com にアクセス
2. 右上の「+」→「New repository」をクリック
3. Repository name を入力（例: `treasure-hunter`）
4. 「Create repository」をクリック

---

### ステップ2: 4つのファイルをアップロード

GitHubの画面で「uploading an existing file」をクリックし、以下の4ファイルをドラッグ&ドロップ：

#### ファイル1: `app.py`
- ダウンロードした `streamlit_app_complete.py` を
- `app.py` にリネームしてアップロード

#### ファイル2: `sector_master.py`
- そのままアップロード

#### ファイル3: `requirements.txt`
- そのままアップロード

#### ファイル4: `README.md`
- そのままアップロード

**「Commit changes」をクリック**

---

### ステップ3: Streamlit Cloudでデプロイ

1. https://share.streamlit.io にアクセス
2. 「Sign in with GitHub」でログイン
3. 「New app」をクリック

4. **設定を入力:**
   ```
   Repository: あなたのGitHub名/treasure-hunter
   Branch: main
   Main file path: app.py
   ```

5. **「Deploy!」をクリック**

6. **5-10分待つ**

7. **完成！新しいURLが発行されます:**
   ```
   https://あなたの名前-treasure-hunter.streamlit.app
   ```

---

## 🎨 推奨リポジトリ名

以下のような名前がおすすめです：

- `treasure-hunter` （お宝ハンター）
- `stock-treasure` （株のお宝）
- `chart-pirate` （チャート海賊）
- `onepiece-stock` （ワンピース株）
- `treasure-finder` （お宝発掘者）

---

## ✅ 完了後の確認

新しいアプリが起動したら：

1. ✅ タイトルが「探せ！この世のすべてをそこに置いてきた」になっている
2. ✅ アイコンが 🏴‍☠️ になっている
3. ✅ サブタイトルが「世はまさに、大海賊時代！」になっている
4. ✅ マニュアルページが海賊風になっている

---

## 📱 URLの共有

完成したら、以下のようにURLを共有できます：

```
🏴‍☠️ お宝発掘ツール
https://あなたの名前-treasure-hunter.streamlit.app

世はまさに、大海賊時代！
有望銘柄を自動で発掘するログポースだ！
```

---

## 🔄 既存アプリとの違い

| 項目 | 既存アプリ | 新アプリ |
|------|----------|---------|
| 名前 | 株式スクリーニングツール | お宝発掘ツール |
| テーマ | ビジネス | ワンピース |
| URL | stock-scanner | treasure-hunter |
| アイコン | 📈 | 🏴‍☠️ |

**両方を並行して運用することも可能です！**

---

## 💡 おすすめの運用方法

### パターンA: 新アプリのみ使用
- 既存アプリは削除またはそのまま
- 新アプリを正式版として使用

### パターンB: 両方を並行運用
- 既存アプリ: 真面目なビジネス用途
- 新アプリ: 楽しく使う個人用途

### パターンC: 段階的移行
- 既存アプリを残しつつ
- 新アプリを試験運用
- 問題なければ新アプリに一本化

---

## 🎉 完成！

新しい「お宝発掘ツール」の誕生です！

**世はまさに、大海賊時代！🏴‍☠️**

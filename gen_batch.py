# Neuronpedia generate API でattribution graphを量産する
# 礼儀: 逐次実行 + 各リクエスト間6秒待機。連続4失敗で中断。
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://www.neuronpedia.org/api/graph/generate"
MODEL = "gemma-2-2b"
OUT = Path(__file__).parent / "data" / "batch"
MANIFEST = Path(__file__).parent / "data" / "manifest.csv"
SLEEP_OK = 20
SLEEP_FAIL = 60
SLEEP_429 = 600  # レート制限時は10分待つ
MAX_429_RETRIES = 36  # 計6時間まで粘る(日次クォータ対策)

# (category, prompt, expected_answers)  expected=[] は「正解が存在しない」反事実枠
PROMPTS = [
    # A: 事実想起(首都)
    ("A", "The capital of France is", ["paris"]),
    ("A", "The capital of Japan is", ["tokyo"]),
    ("A", "The capital of Italy is", ["rome"]),
    ("A", "The capital of Germany is", ["berlin"]),
    ("A", "The capital of Spain is", ["madrid"]),
    ("A", "The capital of Russia is", ["moscow"]),
    ("A", "The capital of China is", ["beijing"]),
    ("A", "The capital of Egypt is", ["cairo"]),
    ("A", "The capital of Canada is", ["ottawa"]),
    ("A", "The capital of Australia is", ["canberra"]),
    # B: 反事実・架空(幻覚誘発枠)
    ("B", "The capital of Atlantis is", []),
    ("B", "The capital of the Moon is", []),
    ("B", "The capital of Narnia is", []),
    ("B", "The king of the United States is", []),
    ("B", "The fourth side of the triangle is", []),
    ("B", "The capital of the Pacific Ocean is", []),
    ("B", "The president of Antarctica is", []),
    ("B", "The national language of Mars is", []),
    ("B", "The queen of Texas is", []),
    ("B", "The capital of the number seven is", []),
    # C: 多段推論(都市→州→州都)
    ("C", "Fact: The capital of the state containing Dallas is", ["austin"]),
    ("C", "Fact: The capital of the state containing Miami is", ["tallahassee"]),
    ("C", "Fact: The capital of the state containing Chicago is", ["springfield"]),
    ("C", "Fact: The capital of the state containing Seattle is", ["olympia"]),
    ("C", "Fact: The capital of the state containing Philadelphia is", ["harrisburg"]),
    ("C", "Fact: The capital of the state containing Los Angeles is", ["sacramento"]),
    ("C", "Fact: The capital of the state containing Las Vegas is", ["carson"]),
    ("C", "Fact: The capital of the state containing Memphis is", ["nashville"]),
    ("C", "Fact: The capital of the state containing Detroit is", ["lansing"]),
    ("C", "Fact: The capital of the state containing Baltimore is", ["annapolis"]),
    # D: 対義語
    ("D", "The opposite of hot is", ["cold"]),
    ("D", "The opposite of big is", ["small", "little"]),
    ("D", "The opposite of up is", ["down"]),
    ("D", "The opposite of fast is", ["slow"]),
    ("D", "The opposite of dark is", ["light", "bright"]),
    ("D", "The opposite of happy is", ["sad", "unhappy"]),
    ("D", "The opposite of old is", ["young", "new"]),
    ("D", "The opposite of wet is", ["dry"]),
    ("D", "The opposite of hard is", ["soft", "easy"]),
    ("D", "The opposite of tall is", ["short"]),
    # F: Fact:付き首都(Aと同内容、フレーミングだけ違う)
    ("F", "Fact: The capital of France is", ["paris"]),
    ("F", "Fact: The capital of Japan is", ["tokyo"]),
    ("F", "Fact: The capital of Italy is", ["rome"]),
    ("F", "Fact: The capital of Germany is", ["berlin"]),
    ("F", "Fact: The capital of Spain is", ["madrid"]),
    ("F", "Fact: The capital of Russia is", ["moscow"]),
    ("F", "Fact: The capital of China is", ["beijing"]),
    ("F", "Fact: The capital of Egypt is", ["cairo"]),
    ("F", "Fact: The capital of Canada is", ["ottawa"]),
    ("F", "Fact: The capital of Australia is", ["canberra"]),
    # E: 算術(文章)
    ("E", "Two plus three equals", ["five", "5"]),
    ("E", "Four plus four equals", ["eight", "8"]),
    ("E", "Six plus one equals", ["seven", "7"]),
    ("E", "Five plus two equals", ["seven", "7"]),
    ("E", "Three plus three equals", ["six", "6"]),
    ("E", "Nine plus one equals", ["ten", "10"]),
    ("E", "Seven plus two equals", ["nine", "9"]),
    ("E", "One plus five equals", ["six", "6"]),
    ("E", "Eight plus two equals", ["ten", "10"]),
    ("E", "Four plus three equals", ["seven", "7"]),
    # G/H: 言い換え実験(末尾追加で既存インデックスを保持。回路は内容を追うか言い回しを追うか)
    ("G", "France's capital city is", ["paris"]),
    ("G", "Japan's capital city is", ["tokyo"]),
    ("G", "Germany's capital city is", ["berlin"]),
    ("G", "Egypt's capital city is", ["cairo"]),
    ("H", "The city that serves as the capital of France is", ["paris"]),
    ("H", "The city that serves as the capital of Japan is", ["tokyo"]),
    ("H", "The city that serves as the capital of Germany is", ["berlin"]),
    ("H", "The city that serves as the capital of Egypt is", ["cairo"]),
    # K: 難問首都(2Bモデルが間違えやすいマイナー国。正誤で構造比較する幻覚実験用)
    ("K", "Fact: The capital of Burkina Faso is", ["ouagadougou"]),
    ("K", "Fact: The capital of Kazakhstan is", ["astana", "nur-sultan"]),
    ("K", "Fact: The capital of Myanmar is", ["naypyidaw", "nay"]),
    ("K", "Fact: The capital of Bhutan is", ["thimphu"]),
    ("K", "Fact: The capital of Mongolia is", ["ulaanbaatar", "ulan"]),
    ("K", "Fact: The capital of Madagascar is", ["antananarivo"]),
    ("K", "Fact: The capital of Uruguay is", ["montevideo"]),
    ("K", "Fact: The capital of Cambodia is", ["phnom"]),
    ("K", "Fact: The capital of Slovakia is", ["bratislava"]),
    ("K", "Fact: The capital of Slovenia is", ["ljubljana"]),
    ("K", "Fact: The capital of Latvia is", ["riga"]),
    ("K", "Fact: The capital of Estonia is", ["tallinn"]),
    # L: 逆方向(reversal curse検証。Kと同じ知識を逆から問う)
    ("L", "Fact: The country whose capital is Ouagadougou is", ["burkina"]),
    ("L", "Fact: The country whose capital is Astana is", ["kazakhstan"]),
    ("L", "Fact: The country whose capital is Naypyidaw is", ["myanmar", "burma"]),
    ("L", "Fact: The country whose capital is Thimphu is", ["bhutan"]),
    ("L", "Fact: The country whose capital is Ulaanbaatar is", ["mongolia"]),
    ("L", "Fact: The country whose capital is Antananarivo is", ["madagascar"]),
    ("L", "Fact: The country whose capital is Montevideo is", ["uruguay"]),
    ("L", "Fact: The country whose capital is Phnom Penh is", ["cambodia"]),
    ("L", "Fact: The country whose capital is Bratislava is", ["slovakia"]),
    ("L", "Fact: The country whose capital is Ljubljana is", ["slovenia"]),
    ("L", "Fact: The country whose capital is Riga is", ["latvia"]),
    ("L", "Fact: The country whose capital is Tallinn is", ["estonia"]),
    # M: 文字数カウント(LLMが構造的に苦手な領域。正誤を割って幻覚の構造比較をする3度目の挑戦)
    ("M", "Fact: The number of letters in the word 'cat' is", ["3", "three"]),
    ("M", "Fact: The number of letters in the word 'sun' is", ["3", "three"]),
    ("M", "Fact: The number of letters in the word 'tree' is", ["4", "four"]),
    ("M", "Fact: The number of letters in the word 'book' is", ["4", "four"]),
    ("M", "Fact: The number of letters in the word 'house' is", ["5", "five"]),
    ("M", "Fact: The number of letters in the word 'water' is", ["5", "five"]),
    ("M", "Fact: The number of letters in the word 'banana' is", ["6", "six"]),
    ("M", "Fact: The number of letters in the word 'flower' is", ["6", "six"]),
    ("M", "Fact: The number of letters in the word 'kitchen' is", ["7", "seven"]),
    ("M", "Fact: The number of letters in the word 'morning' is", ["7", "seven"]),
    ("M", "Fact: The number of letters in the word 'computer' is", ["8", "eight"]),
    ("M", "Fact: The number of letters in the word 'mountain' is", ["8", "eight"]),
    # N: few-shot文字数カウント(末尾空白で数字top-1を強制。スクリーニング済み: 正答6/誤答6)
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'sun' has ", ["3"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'tree' has ", ["4"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'book' has ", ["4"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'house' has ", ["5"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'water' has ", ["5"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'banana' has ", ["6"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'flower' has ", ["6"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'kitchen' has ", ["7"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'morning' has ", ["7"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'computer' has ", ["8"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'mountain' has ", ["8"]),
    ("N", "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word 'elephant' has ", ["8"]),
]


def post_generate(prompt: str, slug: str) -> dict:
    body = json.dumps({"prompt": prompt, "modelId": MODEL, "slug": slug}).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def download(url: str, dest: Path) -> int:
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(PROMPTS)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = {}
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8", newline="") as f:
            rows = {r["slug"]: r for r in csv.DictReader(f)}

    consecutive_fail = 0
    count_429 = 0
    done = 0
    queue = list(enumerate(PROMPTS))
    while queue:
        if done >= limit:
            break
        i, (cat, prompt, expected) = queue.pop(0)
        slug = f"agpoc-{cat.lower()}{i:02d}-hj328"
        dest = OUT / f"{slug}.json"
        if dest.exists() and slug in rows:
            continue
        if consecutive_fail >= 4:
            print("ABORT: 4 consecutive failures", flush=True)
            break
        try:
            res = post_generate(prompt, slug)
            size = download(res["s3url"], dest)
            rows[slug] = {
                "slug": slug,
                "category": cat,
                "prompt": prompt,
                "expected": "|".join(expected),
                "s3url": res["s3url"],
                "numNodes": res.get("numNodes", ""),
                "numLinks": res.get("numLinks", ""),
            }
            consecutive_fail = 0
            done += 1
            print(f"OK {slug} nodes={res.get('numNodes')} bytes={size}", flush=True)
        except Exception as e:  # noqa: BLE001
            if "429" in str(e):
                count_429 += 1
                if count_429 > MAX_429_RETRIES:
                    print("ABORT: rate limit persists after max retries", flush=True)
                    break
                print(f"RATE-LIMITED {slug}: retry in {SLEEP_429}s ({count_429}/{MAX_429_RETRIES})", flush=True)
                queue.insert(0, (i, (cat, prompt, expected)))  # 同じプロンプトを再試行
                time.sleep(SLEEP_429)
            else:
                consecutive_fail += 1
                print(f"FAIL {slug}: {type(e).__name__}: {e}", flush=True)
                time.sleep(SLEEP_FAIL)
            continue
        # マニフェストは毎回書き出し(中断耐性)
        with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["slug", "category", "prompt", "expected", "s3url", "numNodes", "numLinks"],
            )
            w.writeheader()
            for r in rows.values():
                w.writerow(r)
        time.sleep(SLEEP_OK)

    print(f"DONE: {len(rows)} graphs in manifest", flush=True)


if __name__ == "__main__":
    main()

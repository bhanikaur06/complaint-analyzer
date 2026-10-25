"""
Part 2 - Prediction interface.

Loads the trained emotion classifier and classifies text, either from the
command line or interactively.

Run:  python src/predict.py                       interactive
      python src/predict.py "your complaint"      one-off
      python src/predict.py --demo                built-in examples
"""
import sys
import joblib

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

BAR_WIDTH = 28
COLOR = {"anger": "\033[31m", "frustration": "\033[33m",
         "disappointment": "\033[35m", "anxiety": "\033[36m",
         "satisfaction": "\033[32m", "neutral": "\033[37m"}
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"

DEMO = [
    "There is a gas leak in my kitchen, please send someone urgently.",
    "This is the third time I am writing and nobody has bothered to reply.",
    "Thank you for sorting this out so quickly, I really appreciate it.",
    "I expected much better from a brand like yours. Very disappointed.",
    "Please share the invoice for order 88231 for my records.",
    "This is absolutely unacceptable, you have charged me twice!",
]


def load_model():
    path = C.MODELS / "emotion_final.joblib"
    if not path.exists():
        sys.exit("No trained model found. Run:  python src/train_emotion.py")
    return joblib.load(path)


def classify(model, text):
    probs = model.predict_proba([text])[0]
    pairs = sorted(zip(model.classes_, probs), key=lambda x: -x[1])
    return pairs


def show(text, pairs):
    label, conf = pairs[0]
    col = COLOR.get(label, "")
    print(f"\n{DIM}{text}{RESET}")
    print(f"  {BOLD}{col}{label.upper()}{RESET}  {conf*100:.1f}% confidence")
    for name, p in pairs:
        filled = int(round(p * BAR_WIDTH))
        bar = COLOR.get(name, "") + "#" * filled + RESET + DIM + "." * (BAR_WIDTH - filled) + RESET
        print(f"    {name:<15} {bar} {p*100:>5.1f}%")


def main():
    model = load_model()
    args = sys.argv[1:]

    if args and args[0] == "--demo":
        print(f"{BOLD}Emotion classifier - built-in examples{RESET}")
        for t in DEMO:
            show(t, classify(model, t))
        return

    if args:
        text = " ".join(args)
        show(text, classify(model, text))
        return

    print(f"{BOLD}Emotion classifier{RESET}  {DIM}(blank line or Ctrl+C to quit){RESET}")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            return
        show(text, classify(model, text))


if __name__ == "__main__":
    main()

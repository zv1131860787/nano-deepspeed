import sys

from train_qwen3_zero12 import main


if __name__ == "__main__":
    if "--ds-impl" not in sys.argv:
        sys.argv.extend(["--ds-impl", "official"])
    main()

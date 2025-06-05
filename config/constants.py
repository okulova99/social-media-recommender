from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent 
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Автоматическое создание папок при импорте
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
import sys
import os
sys.path.append(project_root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime

# # log_dir = os.path.join("./logs", "run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# log_dir = os.path.join("./logs", f"directory-test-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

# 修改后
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, "logs")
log_dir = os.path.join(logs_dir, f"directory-test-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

print("log目录", log_dir)
os.makedirs(log_dir, exist_ok=True)
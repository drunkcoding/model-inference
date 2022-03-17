import re
import pandas as pd

with open("power.log", "r") as fp:
    lines = fp.read()

groups = re.findall(r" (\d+) %.* (\d+) / (\d+) W", lines)
print(groups)

utilization, power, total = zip(*groups)

df = pd.DataFrame({"utilization": utilization, "power": power, "total": total})
df.to_csv("power.csv")
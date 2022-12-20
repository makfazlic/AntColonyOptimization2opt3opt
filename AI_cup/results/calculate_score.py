import pandas
import glob
files = glob.glob("*.csv")
score = 0
scores = []
for file in files:
    df = pandas.read_csv(file)
    this_score = df["score"].min()
    rows = df.loc[df["score"] == this_score]
    this_score_path = rows.iloc[-1]["path"]
    this_score_time = rows.iloc[-1]["time"]
    scores.append(
        {
            "file": file,
            "best score": this_score,
            "path": this_score_path,
            "time": this_score_time
        }
        )
    score += this_score

scores = pandas.DataFrame(scores, columns=["file", "best score", "path", "time"])
print(scores)
print("Total score: ", score/len(files))
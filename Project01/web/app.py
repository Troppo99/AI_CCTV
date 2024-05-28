from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]
        if video:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
            video.save(filepath)
            # Call your processing function here
            process_video(filepath)
            return redirect(url_for("results"))
    return render_template("index.html")


@app.route("/results")
def results():
    # Logic to display results
    return render_template("results.html")


if __name__ == "__main__":
    app.run(debug=True)

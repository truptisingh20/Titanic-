from Flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('C:\Users\RAMCHANDRA JI KI JAI\trupti ml\titanic_model.pkl.pkl','rb') as file:
    model = pickle.load(file)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        data = request.form
        gender = int(data.get("gender"))
        age = int(data.get("age"))
        nos = int(data.get("nos"))
        tier = int(data.get("Tier"))
        embarked = int(data.get("Embarked"))
        cabin = int(data.get("Cabin"))
        fare = int(data.get("Fare"))
        npca = int(data.get("npca"))

        #user_input = [tier,gender,age,nos,npca,fare,cabin,embarked]
        user_input = np.array([[tier,gender,age,nos,npca,fare,cabin,embarked]])
        model_output = model.predict(user_input)
        output_user = ''
        if model_output == 0:
            output_user = "Not Survived"
        else:
            output_user = "Survived"
        return render_template('index.html',survived = output_user)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
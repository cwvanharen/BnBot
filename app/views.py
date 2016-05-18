
from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required

from . import app, estimator, target_names


class PredictForm(Form):
    """Fields for Predict"""
    myChoices = ["one", "two", "three"]
    sepal_length = fields.DecimalField('Sepal Length:', places=2, validators=[Required()])
    sepal_width = fields.DecimalField('Sepal Width:', places=2, validators=[Required()])
    petal_length = fields.DecimalField('Petal Length:', places=2, validators=[Required()])
    petal_width = fields.DecimalField('Petal Width:', places=2, validators=[Required()])

    submit = fields.SubmitField('Submit')


@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    prediction = None

    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data
        print(submitted_data)

        # Retrieve values from form
        sepal_length = submitted_data['sepal_length']
        sepal_width = submitted_data['sepal_width']
        petal_length = submitted_data['petal_length']
        petal_width = submitted_data['petal_width']

        # Create array from values
        flower_instance = [sepal_length, sepal_width, petal_length, petal_width]

        my_prediction = estimator.predict(flower_instance)
        # Return only the Predicted iris species
        prediction = target_names[my_prediction].capitalize()

    return render_template('index.html', form=form, prediction=prediction)

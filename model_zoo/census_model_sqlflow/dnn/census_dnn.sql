SELECT *
FROM census_income
TO TRAIN DNNClassifier
WITH model.hidden_units = [10, 20]
COLUMN (
    age, 
    capital_gain, 
    capital_loss, 
    hours_per_week, 
    EMBEDDING(workclass, 16),
    EMBEDDING(education, 16),
    EMBEDDING(martial_status, 16),
    EMBEDDING(occupation, 16),
    EMBEDDING(relationship, 16),
    EMBEDDING(race, 16),
    EMBEDDING(sex, 16),
    EMBEDDING(native_country, 16)
)
LABEL label

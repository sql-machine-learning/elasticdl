SELECT *
FROM census_income
TO TRAIN DNNClassifier
WITH model.hidden_units = [10, 20]
COLUMN (
    NUMERIC(age), 
    NUMERIC(capital_gain), 
    NUMERIC(capital_loss), 
    NUMERIC(hours_per_week), 
    EMBEDDING(HASH(workclass, 64), 16),
    EMBEDDING(HASH(education, 64), 16),
    EMBEDDING(HASH(martial_status, 64), 16),
    EMBEDDING(HASH(occupation, 64), 16),
    EMBEDDING(HASH(relationship, 64), 16),
    EMBEDDING(HASH(race, 64), 16),
    EMBEDDING(HASH(sex, 64), 16),
    EMBEDDING(HASH(native_country, 64), 16)
)
LABEL label

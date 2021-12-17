# Utilisation d'une classe ML en partant du post preprocessing.

class ML:
    def __init__(self, data, target, model, config):
        print()
        print("ML Object created")
        print()
        self.data = data
        self.target = target
        self.train = data[data["split_train_test"] == "train"]
        self.test = data[data["split_train_test"] == "test"]
        self._logistic__penalty = config.get("_logistic__penalty", "l2")
        self._logistic__fit_intercept = config.get("_fit_intercept", True)
        self._logistic__solver = config.get("_fit_solver", "lbfgs")
        self.model_choice = model
        self.model_list = {"logistic", "randomforest"}

    def choose_model(self):
        if self.model_choice.lower() == "logistic":
            print("logistic is loaded")
            self.model = LogisticRegression()

        elif self.model_choice.lower() == "randomforest":
            print("random forest is loaded")
            self.model = RandomForest()

    def define_train_test(self):
        # Ajouter une méthode via train_test_split
        print("Train_test loaded")
        self.x_train = self.train.drop([self.target, "split_train_test"], axis=1)
        self.y_train = self.train[self.target]
        self.x_test = self.test.drop([self.target, "split_train_test"], axis=1)
        self.y_test = self.train[self.target]

    def fit(self):
        model = self.model.fit(self.x_train, self.y_train)
        self.y_pred_train = model.predict(self.x_train)
        self.y_pred_test = model.predict(self.x_test)
        self.y_pred_proba_train = model.predict_proba(self.x_train)[:, 1]
        self.y_pred_proba_test = model.predict_proba(self.x_test)[:, 1]

    def evaluate_auc(self):
        fpr, tpr, thresholds = roc_curve(self.test[self.target], self.y_pred_proba_test)
        self.results = {
            "auc": auc(fpr, tpr)
        }
        print(self.results)



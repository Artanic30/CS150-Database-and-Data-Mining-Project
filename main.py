import pandas as pd
import logging
import numpy as np
import re
import os
import json
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    BaggingClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor, \
    AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, VotingRegressor
from pyspark.sql import SparkSession
import random
from sklearn.svm import SVC, SVR

import copy


class Project:
    def __init__(self, train_filepath, test_filepath, sample_filepath, is_generate_feature=True, is_sample_data=False):
        if is_sample_data:
            self.train_data = pd.read_csv(sample_filepath)
        else:
            self.train_data = pd.read_table(train_filepath)
        self.test_data = pd.read_table(test_filepath)
        self.columns = self.train_data.columns
        self.hash_number = 1
        self.feature_hash = FeatureHasher(n_features=self.hash_number, input_type='string')
        self.spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        self.data_cleaning()
        if not os.path.exists('./data/feature.csv') or is_generate_feature:
            self.feature_generation()

    def data_cleaning(self):
        # remove meaningless part of data
        self.train_data.drop(
            columns=["Row", "First Transaction Time", "Step Start Time", "Correct Transaction Time", "Step End Time",
                     "Step Duration (sec)", "Hints"],
            inplace=True)
        self.columns = self.train_data.columns

        check_list = ["Correct Step Duration (sec)", "Incorrects", "Corrects"]
        data = {}
        for item in check_list:
            data[item] = {
                "std": self.train_data[item].std(),
                "mean": self.compute_mean_spark(self.train_data, item)
            }

        def check_outlier(record):
            dict_re = dict(zip(self.columns, record))
            # item on check list is float 64
            for item in check_list:
                if dict_re[item] and abs(dict_re[item] - data[item]["mean"]) > 10 * data[item]["std"]:
                    return True
            return False

        def check_error(record):
            # wrong data
            dict_re = dict(zip(self.columns, record))
            if dict_re['Correct Step Duration (sec)'] == np.NaN and dict_re['Correct First Attempt'] == 0:
                return True
            elif dict_re['Error Step Duration (sec)'] == np.NaN and dict_re['Incorrects'] == 0:
                return True
            return False

        index = 0
        remove_list = []
        problem_unit = []
        problem_section = []
        oppo_num = []

        for i in self.train_data.values:
            if check_outlier(i) or check_error(i):
                remove_list.append(index)
            else:
                # dealing with problem hierarchy
                dict_re = dict(zip(self.columns, i))
                unit = dict_re["Problem Hierarchy"].split(", ")[0]
                unit = re.sub("Unit ", "", unit)
                section = dict_re["Problem Hierarchy"].split(", ")[1]
                section = re.sub("Section ", "", section)
                problem_unit.append(unit)
                problem_section.append(section)

                if type(dict_re["Opportunity(Default)"]) == str:
                    oppo_num.append(dict_re["Opportunity(Default)"])
                else:
                    oppo_num.append("0")

                # dealing with KC

            index += 1
        write_save_log("number of cleaned record: {}".format(len(remove_list)))
        self.train_data.drop(remove_list, inplace=True)
        self.train_data.drop(columns=["Problem Hierarchy", "Opportunity(Default)", "Error Step Duration (sec)"],
                             inplace=True)
        self.train_data["Problem Unit"] = problem_unit
        self.train_data["Problem Section"] = problem_section
        self.train_data["Opportunity(Default)"] = oppo_num
        # for col in self.train_data.columns:
        #     print(self.train_data[col].describe())
        self.columns = self.train_data.columns

    @staticmethod
    def one_hot_encoder_generator(features_pd, column, data):
        id_ohe = OneHotEncoder()
        id_le = LabelEncoder()
        id_labels = id_le.fit_transform(data[column])

        # id_feature_arr: num(row) * num(unique(id))
        id_feature_arr = id_ohe.fit_transform(pd.DataFrame(id_labels)).toarray()
        id_feature_arr = np.transpose(id_feature_arr)
        for label in id_le.classes_:
            features_pd[label] = id_feature_arr[list(id_le.classes_).index(label)]

    def hash_encoder_generator(self, features_pd, column, data):
        sn_feature = self.feature_hash.fit_transform(data[column]).toarray()
        sn_feature = np.transpose(sn_feature)
        for i in range(self.hash_number):
            features_pd["{}_{}".format(column, i)] = sn_feature[i]

    # @staticmethod
    # def count_intelligent_score(cor_time, cor_first, cor_num, in_cur):
    #     return cor_time * cor_first * (in_cur / cor_num)

    @staticmethod
    def count_intelligent_score(cor_time, cor_first, cor_num, in_cur):
        cor_step_time_score = -cor_time + 100
        if cor_step_time_score < -100:
            cor_step_time_score = -100
        normalize_step = (cor_step_time_score + 100) / 200
        return normalize_step * cor_first * (in_cur / cor_num)

    def compute_mean_spark(self, dataframe, column):
        dataframe[column].to_csv('./data/tem.csv')
        spark_df = self.spark.read.csv('./data/tem.csv')
        spark_df.createOrReplaceTempView("train")
        sqlDF = self.spark.sql("SELECT AVG(_c1) as mean FROM train WHERE _c1 is not Null")
        return json.loads(sqlDF.toJSON().first())['mean']

    def feature_generation(self):
        """
        feature 1: compute the intelligent
        feature 2: compute the difficulty of a problem
        feature 3: sum of difficulty of  knowledge component
        """
        # for col in self.train_data.columns:
        #     print(self.train_data[col].describe())
        write_save_log("start to generate features")
        features_pd = pd.DataFrame()

        # hash encoder encoder ID
        # self.one_hot_encoder_generator(features_pd, "Anon Student Id", self.train_data)
        self.hash_encoder_generator(features_pd, "Anon Student Id", self.train_data)
        write_save_log("ID feature generated")
        # hash encoder Problem Name
        self.hash_encoder_generator(features_pd, "Problem Name", self.train_data)
        write_save_log("Problem Name feature generated")

        # hash encoder Problem Name
        self.hash_encoder_generator(features_pd, "Problem Unit", self.train_data)
        write_save_log("Problem Unit feature generated")

        # hash encoder Problem Name
        self.hash_encoder_generator(features_pd, "Problem Section", self.train_data)
        write_save_log("Problem Section feature generated")

        # directly add problem view
        features_pd["Problem View"] = self.train_data["Problem View"]

        mean_pv = self.compute_mean_spark(features_pd, "Problem View")
        new_pv = []
        for row in features_pd["Problem View"]:
            if not np.isnan(row):
                new_pv.append(row)
            else:
                new_pv.append(mean_pv)
        features_pd.drop(columns=["Problem View"], inplace=True)
        features_pd["Problem View"] = new_pv

        write_save_log("Problem View feature generated")
        # Step Name hash to features

        self.hash_encoder_generator(features_pd, "Step Name", self.train_data)
        write_save_log("Step Name feature generated")

        # next features are precomputed values in train data set
        # person intelligent
        write_save_log("start to generate person intelligent")
        id_unique = self.train_data["Anon Student Id"]
        intelligent_table = dict(zip(id_unique, [0 for i in range(len(id_unique))]))

        id_group = self.train_data.groupby(["Anon Student Id"]).mean()
        for i in range(len(id_group.values)):
            write_save_log("id group process: {}".format(i))
            stu_id = id_group.index[i]
            dict_row = dict(zip(id_group.columns, id_group.values[i]))
            intelligent_table[stu_id] = self.count_intelligent_score(dict_row['Correct Step Duration (sec)'],
                                                                     dict_row['Correct First Attempt'],
                                                                     dict_row['Corrects'], dict_row['Incorrects'])

        problem_group = self.train_data.groupby(["Step Name"]).mean()

        problem_difficulty = {}
        problem_group_cor_first = problem_group["Correct First Attempt"]
        for i in range(len(problem_group_cor_first.index)):
            problem_difficulty[problem_group_cor_first.index[i]] = problem_group_cor_first.values[i]
        problem_difficulty['mean'] = problem_group_cor_first.mean()
        write_save_log("problem difficulty mean : {}".format(problem_difficulty['mean']))
        with open("./data/problem.json", 'w') as f:
            f.write(json.dumps(problem_difficulty))

        unique_KC = self.train_data["KC(Default)"].unique()
        unique_KC_list = []
        for kc in unique_KC:
            if type(kc) == str:
                for true_kc in kc.split("~~"):
                    unique_KC_list.append(true_kc)

        # [correct, total]
        kc_difficulty = dict(zip(unique_KC_list, [[0, 0] for i in range(len(id_unique))]))
        person_intelligent = []
        kc_length = []
        index_count = 0
        for row in self.train_data.values:
            dict_row = dict(zip(self.train_data.columns, row))
            if index_count % 10000 == 0:
                write_save_log("loading feature to dataframe process: {}".format(index_count))

            # processing intelligent_table
            stu_id = dict_row["Anon Student Id"]
            person_intelligent.append(intelligent_table[stu_id])

            # extract kc
            stu_kc = dict_row["KC(Default)"]
            kc_num = 0
            if type(stu_kc) == str:
                kc_num = len(stu_kc.split("~~"))
                for true_kc in stu_kc.split("~~"):
                    if dict_row["Correct First Attempt"] == 1:
                        kc_difficulty[true_kc][0] += 1
                    kc_difficulty[true_kc][1] += 1
            kc_length.append(kc_num)

            index_count += 1

        with open('./data/kc_difficulty.json', 'w') as f:
            re_kc = {}
            for key, value in kc_difficulty.items():
                re_kc[key] = value[0] / value[1]

            kc_difficulty = re_kc

            kc_df = pd.DataFrame({"value": list(kc_difficulty.values())})
            kc_mean = self.compute_mean_spark(kc_df, "value")
            kc_difficulty["mean"] = kc_mean

            f.write(json.dumps(kc_difficulty))

        write_save_log("kc mean: {}".format(kc_mean))
        kc_features = []
        oppo_feature = []
        problem_diff_value = []
        for row in self.train_data.values:
            dict_row = dict(zip(self.train_data.columns, row))
            stu_kc = dict_row["KC(Default)"]
            sum_difficult = 0
            oppo_value = 0
            if type(stu_kc) == str:
                oppo_list = dict_row["Opportunity(Default)"].split("~~")
                for true_kc in stu_kc.split("~~"):
                    oppo_value += int(oppo_list[stu_kc.split("~~").index(true_kc)]) * kc_difficulty[true_kc]
                    sum_difficult += kc_difficulty[true_kc]
                sum_difficult /= len(stu_kc.split("~~"))
                oppo_value /= len(stu_kc.split("~~"))
            else:
                oppo_value = kc_difficulty["mean"]
                sum_difficult = kc_difficulty["mean"]

            # problem difficulty
            problem_diff_value.append(problem_difficulty[dict_row["Step Name"]])

            kc_features.append(sum_difficult)
            oppo_feature.append(oppo_value)

        features_pd["kc difficulty"] = kc_features
        features_pd["kc number"] = kc_length
        features_pd["person_intelligent"] = person_intelligent
        features_pd["oppo value"] = oppo_feature
        features_pd['Problem difficulty'] = problem_diff_value
        write_save_log("feature length: {}".format(len(features_pd.columns)))

        features_pd.to_csv("./data/feature.csv", mode='w', index=False)

        with open('./data/intelligent_table.json', 'w') as f:
            f.write(json.dumps(intelligent_table))

    def predict(self):
        write_save_log("start to predict")
        correct_answer = []
        first_attempt_index = list(self.columns).index('Correct First Attempt')
        for row in self.train_data.values:
            re_cor = row[first_attempt_index]
            if np.isnan(re_cor):
                re_cor = 0
            correct_answer.append(re_cor)

        correct_answer = np.array(correct_answer)
        features = pd.read_csv("./data/feature.csv")

        # for col in features.columns:
        #     print(features[col].describe())

        with open('./data/intelligent_table.json', 'r') as f:
            intelligent_table = json.loads(f.read())

        with open('./data/kc_difficulty.json', 'r') as f:
            kc_table = json.loads(f.read())

        with open('./data/problem.json', 'r') as f:
            problem_table = json.loads(f.read())

        # generate feature for test dataa

        test_features_pd = pd.DataFrame()

        problem_unit = []
        problem_section = []
        problem_values = []
        for row in self.test_data.values:
            dict_re = dict(zip(self.test_data.columns, row))
            unit = dict_re["Problem Hierarchy"].split(", ")[0]
            unit = re.sub("Unit ", "", unit)
            section = dict_re["Problem Hierarchy"].split(", ")[1]
            section = re.sub("Section ", "", section)
            problem_unit.append(unit)
            problem_section.append(section)
            if dict_re["Step Name"] in problem_table.keys():
                problem_values.append(problem_table[dict_re["Step Name"]])
            else:
                problem_values.append(problem_table['mean'])
        self.test_data["Problem Unit"] = problem_unit
        self.test_data["Problem Section"] = problem_section

        # one hot encoder ID
        self.hash_encoder_generator(test_features_pd, "Anon Student Id", self.test_data)
        # self.one_hot_encoder_generator(test_features_pd, "Anon Student Id", self.test_data)
        write_save_log("ID feature generated")
        # hash encoder Problem Name
        self.hash_encoder_generator(test_features_pd, "Problem Name", self.test_data)
        write_save_log("Problem Name feature generated")

        # hash encoder Problem Unit
        self.hash_encoder_generator(test_features_pd, "Problem Unit", self.test_data)
        write_save_log("Problem Unit feature generated")

        # hash encoder Problem Section
        self.hash_encoder_generator(test_features_pd, "Problem Section", self.test_data)
        write_save_log("Problem Section feature generated")

        # directly add problem view
        test_features_pd["Problem View"] = self.test_data["Problem View"]

        self.hash_encoder_generator(test_features_pd, "Step Name", self.test_data)

        intel_values = []
        kc_values = []
        test_answer = []
        kc_length = []
        index_count = 0
        remove_list = []
        oppo_feature = []
        for row in self.test_data.values:
            dict_re = dict(zip(self.test_data.columns, row))
            if np.isnan(dict_re["Correct First Attempt"]):
                remove_list.append(index_count)
                test_answer.append(-1)
            else:
                test_answer.append(dict_re["Correct First Attempt"])
            intel_values.append(intelligent_table[dict_re["Anon Student Id"]])

            stu_kc = dict_re["KC(Default)"]
            sum_difficult = 0
            kc_num = 0
            oppo_value = 0
            if type(stu_kc) == str:
                oppo_list = dict_re["Opportunity(Default)"].split("~~")
                kc_num = len(stu_kc.split("~~"))
                for true_kc in stu_kc.split("~~"):
                    oppo_value += int(oppo_list[stu_kc.split("~~").index(true_kc)]) * kc_table[true_kc]
                    sum_difficult += kc_table[true_kc]
                sum_difficult /= len(stu_kc.split("~~"))
            else:
                oppo_value = kc_table["mean"]
                sum_difficult = kc_table["mean"]
            kc_values.append(sum_difficult)
            kc_length.append(kc_num)
            oppo_feature.append(oppo_value)

            index_count += 1

        test_features_pd["kc difficulty"] = kc_values
        test_features_pd["kc number"] = kc_length
        test_features_pd["person_intelligent"] = intel_values
        test_features_pd["oppo value"] = oppo_feature
        test_features_pd['Problem difficulty'] = problem_values
        # test_features_pd.drop(remove_list, inplace=True)

        clf = HistGradientBoostingRegressor(random_state=1,
                                            max_iter=331, loss='least_squares',
                                            learning_rate=0.4,
                                            l2_regularization=0.2)

        clf.fit(features.values, correct_answer)
        res = clf.predict(test_features_pd.values)
        re_res = []
        for i in res:
            if i >= 0.5:
                re_res.append(1)
            else:
                re_res.append(0)

        for i in range(len(re_res)):
            if test_answer[i] == -1:
                test_answer[i] = re_res[i]

        self.test_data.drop(columns=['Correct First Attempt'])
        self.test_data['Correct First Attempt'] = test_answer
        self.test_data.to_csv('./data/final.csv', index=False)

    def train(self):
        write_save_log("start to train")
        correct_answer = []
        first_attempt_index = list(self.columns).index('Correct First Attempt')
        for row in self.train_data.values:
            re_cor = row[first_attempt_index]
            if np.isnan(re_cor):
                re_cor = 0
            correct_answer.append(re_cor)

        correct_answer = np.array(correct_answer)
        features = pd.read_csv("./data/feature.csv")

        # for col in features.columns:
        #     print(features[col].describe())

        with open('./data/intelligent_table.json', 'r') as f:
            intelligent_table = json.loads(f.read())

        with open('./data/kc_difficulty.json', 'r') as f:
            kc_table = json.loads(f.read())

        with open('./data/problem.json', 'r') as f:
            problem_table = json.loads(f.read())

        # generate feature for test dataa

        test_features_pd = pd.DataFrame()

        problem_unit = []
        problem_section = []
        problem_values = []
        for row in self.test_data.values:
            dict_re = dict(zip(self.test_data.columns, row))
            unit = dict_re["Problem Hierarchy"].split(", ")[0]
            unit = re.sub("Unit ", "", unit)
            section = dict_re["Problem Hierarchy"].split(", ")[1]
            section = re.sub("Section ", "", section)
            problem_unit.append(unit)
            problem_section.append(section)
            if dict_re["Step Name"] in problem_table.keys():
                problem_values.append(problem_table[dict_re["Step Name"]])
            else:
                problem_values.append(problem_table['mean'])
        self.test_data["Problem Unit"] = problem_unit
        self.test_data["Problem Section"] = problem_section

        # one hot encoder ID
        self.hash_encoder_generator(test_features_pd, "Anon Student Id", self.test_data)
        # self.one_hot_encoder_generator(test_features_pd, "Anon Student Id", self.test_data)
        write_save_log("ID feature generated")
        # hash encoder Problem Name
        self.hash_encoder_generator(test_features_pd, "Problem Name", self.test_data)
        write_save_log("Problem Name feature generated")

        # hash encoder Problem Unit
        self.hash_encoder_generator(test_features_pd, "Problem Unit", self.test_data)
        write_save_log("Problem Unit feature generated")

        # hash encoder Problem Section
        self.hash_encoder_generator(test_features_pd, "Problem Section", self.test_data)
        write_save_log("Problem Section feature generated")

        # directly add problem view
        test_features_pd["Problem View"] = self.test_data["Problem View"]

        self.hash_encoder_generator(test_features_pd, "Step Name", self.test_data)

        intel_values = []
        kc_values = []
        test_answer = []
        kc_length = []
        index_count = 0
        remove_list = []
        oppo_feature = []
        for row in self.test_data.values:
            dict_re = dict(zip(self.test_data.columns, row))
            if np.isnan(dict_re["Correct First Attempt"]):
                remove_list.append(index_count)
            else:
                test_answer.append(dict_re["Correct First Attempt"])
            intel_values.append(intelligent_table[dict_re["Anon Student Id"]])

            stu_kc = dict_re["KC(Default)"]
            sum_difficult = 0
            kc_num = 0
            oppo_value = 0
            if type(stu_kc) == str:
                oppo_list = dict_re["Opportunity(Default)"].split("~~")
                kc_num = len(stu_kc.split("~~"))
                for true_kc in stu_kc.split("~~"):
                    oppo_value += int(oppo_list[stu_kc.split("~~").index(true_kc)]) * kc_table[true_kc]
                    sum_difficult += kc_table[true_kc]
                sum_difficult /= len(stu_kc.split("~~"))
            else:
                oppo_value = kc_table["mean"]
                sum_difficult = kc_table["mean"]
            kc_values.append(sum_difficult)
            kc_length.append(kc_num)
            oppo_feature.append(oppo_value)

            index_count += 1

        test_features_pd["kc difficulty"] = kc_values
        test_features_pd["kc number"] = kc_length
        test_features_pd["person_intelligent"] = intel_values
        test_features_pd["oppo value"] = oppo_feature
        test_features_pd['Problem difficulty'] = problem_values
        test_features_pd.drop(remove_list, inplace=True)

        parameter_range = {
            "random_state": [i for i in range(0, 40)],
            "max_iter": [i for i in range(100, 500)],
            "loss": ['least_squares', 'least_absolute_deviation', 'poisson'],
            "learning_rate": [0.1 * i for i in range(1, 7)],
            "l2_regularization": [0.1 * i for i in range(1, 10)],
        }
        best_score = 1
        bes_policy = {}
        while best_score > 0.35:
            random_state = {}
            for key, value in parameter_range.items():
                random_state[key] = random.sample(value, 1)
            write_save_log(random_state)

            # clf1 = HistGradientBoostingRegressor()
            # clf2 = AdaBoostRegressor()
            #
            # clf = VotingRegressor(estimators=[('hgb', clf1), ('rf', clf2)], weights=[2, 1])

            clf = HistGradientBoostingRegressor(random_state=random_state["random_state"][0],
                                                max_iter=random_state["max_iter"][0], loss=random_state['loss'][0],
                                                learning_rate=random_state['learning_rate'][0],
                                                l2_regularization=random_state['l2_regularization'][0])

            clf.fit(features.values, correct_answer)

            for i in range(len(test_features_pd.columns)):
                if test_features_pd.columns[i] != features.columns[i]:
                    raise KeyError("feature order error!")

            res = clf.predict(test_features_pd.values)
            re_res = []
            for i in res:
                if i >= 0.5:
                    re_res.append(1)
                else:
                    re_res.append(0)

            re_score = MSER(re_res, test_answer)

            write_save_log("result error: {}".format(re_score))
            if best_score > re_score:
                best_score = re_score
                bes_policy = copy.deepcopy(random_state)
            write_save_log("\nbest policy and score\n" + str(bes_policy))
            write_save_log(str(best_score) + '\n')


def MSER(a, b):
    sum_error = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            sum_error += pow(a[i] - b[i], 2)
    return pow(sum_error / (len(a) - 1), 0.5)


def write_save_log(info):
    logging.info(info)
    print(info)


if __name__ == "__main__":
    logging.basicConfig(filename="./runtime.log", level=logging.DEBUG, filemode='w')
    train = pd.read_table('data/train.csv')
    # for col in train.columns:
    #     print(train[col].describe())
    # ["Anon Student Id", "Problem Hierarchy", "Problem Name", "Problem View", "Step Name", "Correct Step Duration (sec)", "Correct First Attempt", "Incorrects", "KC(Default)", "Opportunity(Default)"]
    p = Project('data/train.csv', 'data/test.csv', 'data/sample.csv', is_generate_feature=False, is_sample_data=False)
    p.predict()
    a = 1

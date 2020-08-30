import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.feature_selection import mutual_info_regression
import logging
root = os.path.abspath(os.path.dirname('__file__'))
sys.path.append(root)



def dump_norm_ind(sMax, sMin, save_path):
    """Dump normalization indicators to local disk.

    Arguments:
        \nsMax: [Float list, array, series or dataframe] Maximum values of predictors and predicted target.
        \nsMin: [Float list, array, series or dataframe] Minimum values of predictors and predicted target.
        \nsave_path: [String] The save path.

    """
    sMax = pd.DataFrame(sMax, columns=['series_max'])
    sMin = pd.DataFrame(sMin, columns=['series_min'])
    norm_ind = pd.concat([sMax, sMin], axis=1)
    norm_ind.to_csv(save_path+"norm_id.csv")


def PCA_transform(X, y, n_components):
    """Dimension reduction based on principle component analysis(PCA).

    Argument:
        \nX: [Float dataframe] Predictors.
        \ny: [Float series] Predicted target.
        \nn_component: [String or int] The number of retained predictors. If string, only 'mle' (guessed by MLE method) is optional.

    """
    # logger.info('X.shape={}'.format(X.shape))
    # logger.info('y.shape={}'.format(y.shape))
    # logger.info('X contains Nan:{}'.format(X.isnull().values.any()))
    # logger.info("Input features before PAC:\n{}".format(X))
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    pca_X = pca.transform(X)
    columns = []
    for i in range(1, pca_X.shape[1]+1):
        columns.append('X'+str(i))
    pca_X = pd.DataFrame(pca_X, columns=columns)
    logger.info("pca_X.shape={}".format(pca_X.shape))
    print(pca_X)
    print(y)
    pca_samples = pd.concat([pca_X, y], axis=1)
    return pca_samples


def gen_direct_samples(input_df, output_df, lags_dict, lead_time):
    """Generate direct samples. 

    Arguments:
        \ninput_df: [float dataframe] Predictors source.
        \noutput_df: [float dataframe] Predicted target source.
        \nlags_dict: [dict] Lagged time for each column in 'input_df'.
        \nlead_time: [int] leading time.

    """
    input_columns = list(input_df.columns)
    # Get the number of input features
    signals_num = input_df.shape[1]
    # Get the data size
    data_size = input_df.shape[0]
    # Compute the samples size
    max_lag = max(lags_dict.values())
    # logger.info('max lag:{}'.format(max_lag))
    samples_size = data_size-max_lag
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags_dict.values())):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # logger.info('Samples columns:{}'.format(samples_cols))
    # Generate input colmuns for each input feature
    samples = pd.DataFrame()
    for i in range(signals_num):
        # Get one input feature
        one_in = (input_df[input_columns[i]]).values  # subsignal
        lag = lags_dict[input_columns[i]]
        # logger.info('lag:{}'.format(lag))
        oness = pd.DataFrame()  # restor input features
        for j in range(lag):
            x = pd.DataFrame(one_in[j:data_size-(lag-j)],
                             columns=['X' + str(j + 1)])
            x = x.reset_index(drop=True)
            oness = pd.concat([oness, x], axis=1, sort=False)
        # logger.info("oness:\n{}".format(oness))
        oness = oness.iloc[oness.shape[0]-samples_size:]
        oness = oness.reset_index(drop=True)
        samples = pd.concat([samples, oness], axis=1, sort=False)
    # Get the target
    target = (output_df.values)[max_lag+lead_time-1:]
    target = pd.DataFrame(target, columns=['Y'])
    # logger.info('Target=\n{}'.format(target))
    # Concat the features and target
    samples = samples[:samples.shape[0]-(lead_time-1)]
    samples = samples.reset_index(drop=True)
    samples = pd.concat([samples, target], axis=1)
    samples = pd.DataFrame(samples.values, columns=samples_cols)
    return samples


def get_last_direct_sample(input_df, output_df, lags_dict, lead_time):
    """Get last sample of direct samples.

    Arguments:
        \ninput_df: [float dataframe] Predictors source.
        \noutput_df: [float dataframe] Predicted target source.
        \nlags_dict: [dict] Lagged time for each column in 'input_df'.
        \nlead_time: [int] Leading time.

    """
    samples = gen_direct_samples(input_df, output_df, lags_dict, lead_time)
    last_sample = samples.iloc[samples.shape[0]-1:]
    return last_sample


def gen_autoreg_samples(signal, lag, lead_time):
    """Generate samples from a signal (autoregressive pattern).

    Arguments:
        \nsignal: [float dataframe] The source signal.
        \nlag: [int] Lagged time.
        \nlead_time: [int] Leading time.

    """
    if type(signal) == pd.DataFrame or type(signal) == pd.Series or type(signal) == list:
        nparr = np.array(signal)
    # Create an empty pandas Dataframe
    samples = pd.DataFrame()
    # Generate input series based on lag and add these series to full dataset
    for i in range(lag):
        x = pd.DataFrame(
            nparr[i:signal.shape[0] - (lag - i)], columns=['X' + str(i + 1)])
        x = x.reset_index(drop=True)
        samples = pd.concat([samples, x], axis=1, sort=False)
    # Generate label data
    target = pd.DataFrame(nparr[lag+lead_time-1:], columns=['Y'])
    target = target.reset_index(drop=True)
    samples = samples[:samples.shape[0]-(lead_time-1)]
    samples = samples.reset_index(drop=True)
    # Add labled data to full_data_set
    samples = pd.concat([samples, target], axis=1, sort=False)
    return samples


def get_last_autoreg_sample(signal, lag, lead_time):
    """Get the last sample of signal samples.

    Arguments:
        \nsignal: [float dataframe] The source signal.
        \nlag: [int] Lagged time.
        \nlead_time: [int] Leading time.

    """
    samples = gen_autoreg_samples(signal, lag, lead_time)
    last_sample = samples.iloc[samples.shape[0]-1:]
    return last_sample


def gen_direct_pcc_samples(input_df, output_df, pre_times, lead_time, lags_dict, mode):
    """Generate direct samples based on Pearson coefficient correlation (PCC).

    Arguments:
        \ninput_df: [float dataframe] Predictors source.
        \noutput_df: [float dataframe] Predicted target source.
        \npre_times: [int] The maximum lagged times for selecting optimal lagged time.
        \nlags_dict: [dict] The optimal lagged time.

    """
    input_columns = list(input_df.columns)
    pcc_lag = pre_times+lead_time
    pre_cols = []
    for i in range(1, pre_times+1):
        pre_cols.append("X"+str(i))
    # logger.info("Previous columns of lagged months:\n{}".format(pre_cols))
    # logger.info('PCC mode={}'.format(mode))
    if mode == 'global':
        cols = []
        for i in range(1, pre_times*input_df.shape[1]+1):
            cols.append('X'+str(i))
        cols.append('Y')
        # logger.info("columns of lagged months:\n{}".format(cols))
        input_predictors = pd.DataFrame()
        for col in input_columns:
            # logger.info("Perform subseries:{}".format(col))
            subsignal = np.array(input_df[col])
            inputs = pd.DataFrame()
            for k in range(pcc_lag):
                x = pd.DataFrame(
                    subsignal[k:subsignal.size-(pcc_lag-k)], columns=["X"+str(k+1)])
                x = x.reset_index(drop=True)
                inputs = pd.concat([inputs, x], axis=1)
            pre_inputs = inputs[pre_cols]
            input_predictors = pd.concat(
                [input_predictors, pre_inputs], axis=1)

        # logger.info("Input predictors:\n{}".format(input_predictors.head()))
        target = output_df[pcc_lag:]
        target = target.reset_index(drop=True)
        samples = pd.concat([input_predictors, target], axis=1)
        samples = pd.DataFrame(samples.values, columns=cols)
        # logger.info("Inputs and output:\n{}".format(samples.head()))
        corrs = samples.corr(method="pearson")
        # logger.info("Entire pearson correlation coefficients:\n{}".format(corrs))
        corrs = (corrs['Y']).iloc[0:corrs.shape[0]-1]
        logger.info("Pearson correlation coefficients:\n{}".format(corrs))
        orig_corrs = abs(corrs.squeeze())
        orig_corrs = orig_corrs.sort_values(ascending=False)
        # logger.info("Descending pearson coefficients:\n{}".format(orig_corrs))
        # logger.info('Lags_dict.valus={}'.format(list(lags_dict.values())))
        PACF_samples_num = sum(list(lags_dict.values()))
        selected_corrs = orig_corrs[:PACF_samples_num]
        selected_cols = list(selected_corrs.index.values)
        # logger.info("Selected columns:\n{}".format(selected_cols))
        selected_cols.append('Y')
        samples = samples[selected_cols]
        # logger.info("Selected samples:\n{}".format(samples))
        columns = []
        for i in range(0, samples.shape[1]-1):
            columns.append("X"+str(i+1))
        columns.append("Y")
        samples = pd.DataFrame(samples.values, columns=columns)
        return samples
    elif mode == 'local':
        target = output_df[pcc_lag:]
        target = target.reset_index(drop=True)
        input_predictors = pd.DataFrame()
        cols = []
        for i in range(1, pre_times+1):
            cols.append('X'+str(i))
        cols.append('Y')
        for col in input_columns:
            # logger.info("Perform subseries:{}".format(col))
            lag = lags_dict[col]
            # logger.info('lag={}'.format(lag))
            subsignal = np.array(input_df[col])
            inputs = pd.DataFrame()
            for k in range(pcc_lag):
                x = pd.DataFrame(
                    subsignal[k:subsignal.size-(pcc_lag-k)], columns=["X"+str(k+1)])
                x = x.reset_index(drop=True)
                inputs = pd.concat([inputs, x], axis=1)
            pre_inputs = inputs[pre_cols]
            samples = pd.concat([pre_inputs, target], axis=1)
            samples = pd.DataFrame(samples.values, columns=cols)
            # logger.info("Inputs and output:\n{}".format(samples.head()))
            corrs = samples.corr(method="pearson")
            # logger.info( "Entire pearson correlation coefficients:\n{}".format(corrs))
            corrs = (corrs['Y']).iloc[0:corrs.shape[0]-1]
            # logger.info("Pearson correlation coefficients:\n{}".format(corrs))
            orig_corrs = abs(corrs.squeeze())
            orig_corrs = orig_corrs.sort_values(ascending=False)
            # logger.info("Descending pearson coefficients:\n{}".format(orig_corrs))
            # logger.info('Lags_dict.valus={}'.format(list(lags_dict.values())))
            selected_corrs = orig_corrs[:lag]
            selected_cols = list(selected_corrs.index.values)
            # logger.info("Selected columns:\n{}".format(selected_cols))
            input_samples = samples[selected_cols]
            # logger.info("Selected samples:\n{}".format(input_samples))
            input_predictors = pd.concat(
                [input_predictors, input_samples], axis=1)
        # logger.info("Input predictors:\n{}".format(input_predictors.head()))
        samples = pd.concat([input_predictors, target], axis=1)
        columns = []
        for i in range(0, samples.shape[1]-1):
            columns.append("X"+str(i+1))
        columns.append("Y")
        samples = pd.DataFrame(samples.values, columns=columns)
        return samples


def get_last_direct_pcc_sample(input_df, output_df, pre_times, lead_time, lags_dict, mode):
    """Get the last samples of direct samples
    generated based on Pearson coefficient correlation (PCC).

    Arguments:
        \ninput_df: [float dataframe] Predictors source.
        \noutput_df: [float dataframe] Predicted target source.
        \npre_times: [int] The maximum lagged times for selecting optimal lagged time.
        \nlags_dict: [dict] The optimal lagged time.

    """
    samples = gen_direct_pcc_samples(
        input_df, output_df, pre_times, lead_time, lags_dict, mode)
    last_sample = samples[samples.shape[0]-1:]
    return last_sample


def dump_direct_samples(save_path, train_samples, dev_samples, test_samples):
    """Dump direct samples.

    Arguments:
        \nsave_path: [string].
        \ntrain_samples: [dataframe].
        \ndev_samples: [dataframe].
        \ntest_samples: [dataframe].

    """
    train_samples.to_csv(save_path+'train_samples.csv', index=None)
    dev_samples.to_csv(save_path+'dev_samples.csv', index=None)
    test_samples.to_csv(save_path+'test_samples.csv', index=None)
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / \
        (series_max - series_min) - 1
    dev_samples = 2 * (dev_samples - series_min) / \
        (series_max - series_min) - 1
    test_samples = 2*(test_samples-series_min)/(series_max-series_min)-1
    # logger.info('Save path:{}'.format(save_path))
    # logger.info('The size of training samples:{}'.format(train_samples.shape[0]))
    # logger.info('The size of development samples:{}'.format(dev_samples.shape[0]))
    # logger.info('The size of testing samples:{}'.format(test_samples.shape[0]))
    series_max = pd.DataFrame(series_max, columns=['series_max'])
    series_min = pd.DataFrame(series_min, columns=['series_min'])
    normalize_indicators = pd.concat([series_max, series_min], axis=1)
    normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
    train_samples.to_csv(save_path+'minmax_unsample_train.csv', index=None)
    dev_samples.to_csv(save_path+'minmax_unsample_dev.csv', index=None)
    test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)


def dump_multicomponent_samples(save_path, signal_id, train_samples, dev_samples, test_samples):
    """Dump multi-component samples [autoregressive pattern].

    Arguments:
        \nsave_path: [string].
        \ntrain_samples: [dataframe].
        \ndev_samples: [dataframe].
        \ntest_samples: [dataframe].

    """
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    train_samples = 2 * (train_samples - series_min) / \
        (series_max - series_min) - 1
    dev_samples = 2 * (dev_samples - series_min) / \
        (series_max - series_min) - 1
    test_samples = 2*(test_samples-series_min)/(series_max-series_min)-1
    series_max = pd.DataFrame(series_max, columns=['series_max'])
    series_min = pd.DataFrame(series_min, columns=['series_min'])
    normalize_indicators = pd.concat([series_max, series_min], axis=1)
    normalize_indicators.to_csv(
        save_path+'norm_unsample_id_s'+str(signal_id)+'.csv')
    train_samples.to_csv(save_path+'minmax_unsample_train_s' +
                         str(signal_id)+'.csv', index=None)
    dev_samples.to_csv(save_path+'minmax_unsample_dev_s' +
                       str(signal_id)+'.csv', index=None)
    test_samples.to_csv(save_path+'minmax_unsample_test_s' +
                        str(signal_id)+'.csv', index=None)


import os
import argparse
import pandas as pd


if __name__ == "__main__":
    
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="CATS", choices=['CATS','influx'])
    parser.add_argument("--data_csv", type=str, default="./CATS/data.csv")
    parser.add_argument("--anomalies_csv", type=str, default="./CATS/metadata.csv")
    parser.add_argument("--sensors", type=str, default="temperature,volume,humidity")
    parser.add_argument("--model", type=str, default="blade")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--train_eval_split", type=float, default=0.6)
    
    args = parser.parse_args()
    
    
    if args.source == "CATS":
        
        # read in CATS data
        # skip inital rows without anomalies
        # and filter columns - entries in bfo2 & bso2 are root causes for a number of anomalies, and cso1 is affected by both
        df = pd.read_csv(args.data_csv, usecols=['timestamp','bfo2','bso2','cso1'], skiprows=range(1,950401),
                         index_col='timestamp', parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
        
        # read in anomalies, for selected columns only
        anomaly_ranges = pd.read_csv(args.anomalies_csv, parse_dates=['start_time','end_time'], date_format='%Y-%m-%d %H:%M:%S')
        anomaly_ranges = anomaly_ranges[anomaly_ranges['root_cause'].isin(['bfo2','bso2'])]
        
        # set output columns
        columns = ['bfo2','bso2','cso1']
        
    else:
    
        # read in influxdb export
        df = pd.read_csv(args.data_csv, usecols=['_time','_field','_value'], skiprows=3,
                         parse_dates=['_time'], date_format='%Y-%m-%dT%H:%M:%S.%fZ')
                    
        # group individual sensor readings by time
        df = df.pivot(index="_time", columns="_field", values="_value")
        
        # read in anomalies
        anomaly_ranges = pd.read_csv(args.anomalies_csv, parse_dates=['start_time','end_time'], date_format='%Y-%m-%d %H:%M:%S')
        
        # filter sensors
        columns = args.sensors.split(',')
        
        
    # set label to true for anomalies in given ranges
    df['anomaly'] = 0
    for a in anomaly_ranges.itertuples(index=False):
        df.loc[a.start_time:a.end_time, 'anomaly'] = 1
    
    if args.steps != 1:
        # select every nth row only
        df = df.iloc[::args.steps]
    
    # split into train and eval data
    split = int(len(df)*args.train_eval_split)
    train_data = df.iloc[:split, :]
    eval_data = df.iloc[split:, :]
    
    # save to separate csv
    output_dir = f'./datasets/{args.model}'
    os.makedirs(output_dir, exist_ok=True)
    
    train_data.to_csv(f'{output_dir}/train.csv', columns=columns, index=False, header=False)
    train_data.to_csv(f'{output_dir}/train_labels.csv', columns=['anomaly'], index=False, header=False)
    
    eval_data.to_csv(f'{output_dir}/eval.csv', columns=columns, index=False, header=False)
    eval_data.to_csv(f'{output_dir}/eval_labels.csv', columns=['anomaly'], index=False, header=False)
    
    print(f"Created train and eval datasets for '{args.model}' model from {args.source} under '{output_dir}'.")
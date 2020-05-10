def prep_data(df):
    
    # df = df.assign(hw = df["Height"] * df["Width"] )
    # df = df.assign(all_l = (df["Length1"] * df["Length2"] * df["Length3"]).mean() )
    # X = df[["hw", "all_l"]].values

    X = df[["Height", "Width", "Length3", "Length2"]].values
    y = df["Weight"].values
    
    return X, y
import pandas as pd
import math

# name of ws list produced by zip2ws
zip2ws_csv = 'all-zip2ws.csv'
not_a_state = ['AA', 'AP', 'GU', 'AS', 'PQ', 'FM', 'MP', 'MH', 'PR', 'AE']


def get_unique_ws(ws_df, ws_id='cws_id'):
    uni_name = []
    uni_index = []
    for i, w in enumerate(ws_df[ws_id]):
        if w not in uni_name:
            uni_name.append(w)
            uni_index.append(i)

    uni_ws_df = ws_df.iloc[uni_index]
    return uni_ws_df


# find closest weather station, if non, remove row
def find_closest_ws(ws_df):
    # ws dist start at 17 (id, name, dist)
    closest_ws = []
    for r in range(1000): # len(df)):
        temp = []
        if ws_df['state'][r] not in not_a_state:
            for i in range(17, len(ws_df.iloc[r]), 3):
                if not math.isnan(ws_df.iloc[r][i]):
                    temp.append(ws_df.iloc[r][i])
                else:
                    # exit for loop if nan
                    break
        # else:
        #    print('not a state: {}, {}'.format(ws_df['city'][r],
        #                                       ws_df['state'][r]))

        # if a ws exists for this zip, find min distance
        if len(temp) > 0:
            # find location of min dist in df
            df_dist_loc = temp.index(min(temp))*3 + 17
            df_id_loc = df_dist_loc - 2
            name = df_dist_loc - 1
            closest_ws.append((ws_df.iloc[r][df_id_loc], ws_df.iloc[r][name],
                              ws_df.iloc[r][df_dist_loc]))
        else:
            closest_ws.append(('none', 'none', float('nan')))

    # build new df with zip, city, state, id, name, dist
    new_data_index = []
    for i in range(len(closest_ws)):
        if not math.isnan(closest_ws[i][2]):
            # print("append: {}".format(closest_ws[i][2]))
            new_data_index.append(i)

    ws_close = ws_df.loc[new_data_index, ['zip', 'city', 'state']]
    ws_close['cws_id'] = [closest_ws[i][0] for i in new_data_index]
    ws_close['cws_name'] = [closest_ws[i][1] for i in new_data_index]
    ws_close['cws_dist'] = [closest_ws[i][2] for i in new_data_index]
    return ws_close


if __name__ == '__main__':
    print('running as main...')
    ws_df = pd.read_csv(zip2ws_csv, converters={'zip': lambda x: str(x)},
                        low_memory=False)
    # unique = get_unique_ws(ws_df)
    # closest ws
    # cws_df = find_closest_ws(ws_df)
    # cws_df.to_csv('test-cws.csv', index=False)
    cws_df = pd.read_csv('test-cws.csv', converters={'zip': lambda x: str(x)},

                        low_memory=False)
    uni_cws = get_unique_ws(cws_df)
    uni_cws.to_csv('uni-cws.csv', index=False)
    print("len cws: {} len uni-cws: {}".format(len(cws_df), len(uni_cws)))

import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px  # pip install plotly-express
import streamlit as st
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.cluster import KMeans
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
from streamlit_gsheets import GSheetsConnection



# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

# --- USER AUTHENTICATION ---
names = ["admin satu", "admin dua"]
usernames = ["admin", "admin1"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef", cookie_expiry_days=10)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    url = "https://docs.google.com/spreadsheets/d/1URkvLnGyu9IH08DTOl-hlX_u3-cJB7JnR_vdsgR2_7o/edit?usp=sharing"
    conn = st.experimental_connection("gsheets", type=GSheetsConnection)

    df = conn.read(spreadsheet=url)

# ---- MAINPAGE ----
    st.title(":bar_chart: Sales Dashboard")
    st.markdown("##")

    # TOP KPI's
    total_sales = int(df["Sales"].sum())
    average_sale_by_transaction = round(df["Sales"].mean(), 2)

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Sales:")
        st.subheader(f"US $ {total_sales:,}")
    with right_column:
        st.subheader("Average Sales Per Transaction:")
        st.subheader(f"US $ {average_sale_by_transaction}")

    st.markdown("""---""")

    #Line Chart
    col1, col2 = st.columns((2))
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # Getting the min and max date 
    startDate = pd.to_datetime(df["Order Date"]).min()
    endDate = pd.to_datetime(df["Order Date"]).max()

    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))

    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

    df["month_year"]=df["Order Date"].dt.to_period("M")
    st.subheader ('Time Series Analysis')

    linechart = pd.DataFrame(df.groupby(df["month_year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
    fig2 = px.line(linechart, x="month_year", y="Sales", labels={"Sales":"Amount"}, height=500, width=1000,template="gridon")
    st.plotly_chart(fig2,use_container_width=True)

    #RFM
    col1,col2 = st.columns((2))
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    #RFM
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Penjualan"] = df["Sales"]*df["Quantity"]
    df["Recency"] = df["Order Date"]
    df["Frequency"] = df["Customer ID"]
    df["Monetary"] = df["Penjualan"]

    count_row_with_nan = df.isnull().any(axis=1).sum()

    print ('Count rows with NaN: ' + str(count_row_with_nan))

    count_neg_sales = (df['Sales'] < 0).sum()
    count_neg_quantity = (df['Quantity'] < 0).sum()



    neg_quantity_and_price = df[(df['Quantity'] < 0) |
                                    (df['Sales'] < 0)].index
    df.drop(neg_quantity_and_price, inplace=True)

    df['Customer ID'].count()

    df['Customer ID'].nunique()

    count_row_with_nan = df.isnull().any(axis=1).sum()
    print ('Count rows with NaN: ' + str(count_row_with_nan))
    count_neg_sales = (df['Sales'] < 0).sum()
    count_neg_quantity = (df['Quantity'] < 0).sum()

    import datetime as dt
    startDate = pd.to_datetime(df["Order Date"]).min()
    endDate = pd.to_datetime(df["Order Date"]).max()
    penjualan_rfm = df.groupby('Customer ID').agg({'Recency': lambda date: (startDate - date.max()).days,
                                                        'Frequency': lambda num: len(num),
                                                        'Monetary': lambda price: price.sum()}).reset_index()

    penjualan_rfm['Recency'] = penjualan_rfm['Recency'].abs()
    count_row_with_nan = penjualan_rfm.isnull().any(axis=1).sum()
    print ('Count rows with NaN: ' + str(count_row_with_nan))
    count_neg_sales = (df['Sales'] < 0).sum()
    count_neg_quantity = (df['Quantity'] < 0).sum()

    cols = list(penjualan_rfm.columns)
    cols.remove("Customer ID")

    penjualan_rfm['Customer ID'].count()

    penjualan_rfm['recency_val'] = pd.qcut(penjualan_rfm['Recency'].rank(method="first"), q=3, labels=['1','2','3'])
    penjualan_rfm['frequency_val'] = pd.qcut(penjualan_rfm['Frequency'].rank(method="first"), q=3, labels=['3','2','1'])
    penjualan_rfm['monetary_val'] = pd.qcut(penjualan_rfm['Monetary'].rank(method="first"), q=3, labels=['3','2','1'])

    # penjualan_rfm['RFM Group'] = penjualan_rfm.recency_val.astype(str) + penjualan_rfm.frequency_val.astype(str)
    penjualan_rfm['RFM Score'] = penjualan_rfm[['recency_val', 'frequency_val', 'monetary_val']].astype(int).sum(axis=1)

    penjualan_rfm = penjualan_rfm.sort_values(by="RFM Score", ascending=True).reset_index(drop=True)

    rfm_level = ['loyal', 'promising','need attention']
    score_cuts = pd.qcut(penjualan_rfm['RFM Score'].rank(method='first'), q=3, labels = rfm_level)
    penjualan_rfm['RFM Level'] = score_cuts.values

    loyal = penjualan_rfm['RFM Level'].value_counts()['loyal']
    promising = penjualan_rfm['RFM Level'].value_counts()['promising']
    need_attention = penjualan_rfm['RFM Level'].value_counts()['need attention']

    range_n_clusters = range(1, 9)
    scoreOfWithinClusterSumOfSquared = []
    data = penjualan_rfm[['Recency','Frequency','Monetary']]
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans = kmeans.fit(data)
        scoreOfWithinClusterSumOfSquared.append(kmeans.inertia_)
    df_elbow_result = pd.DataFrame({'clusters' : range_n_clusters, 'scoreOfWithinClusterSumOfSquared': scoreOfWithinClusterSumOfSquared})
    px.scatter(x="clusters", y="scoreOfWithinClusterSumOfSquared", data_frame=df_elbow_result)

    data = penjualan_rfm[['Recency','Frequency','Monetary']]
    k_means = KMeans(n_clusters = 3, random_state = 42)
    k_means.fit(data)
    labels = k_means.labels_
    penjualan_rfm['ClusterLabels'] = labels

    def kmeans_level(df):
        if (df['ClusterLabels'] == 0):
            return 'Need Attention'
        elif (df['ClusterLabels'] == 1):
            return 'Loyal'
        elif (df['ClusterLabels'] == 2):
            return 'Promising'
    penjualan_rfm['RFM with KMeans Level'] = penjualan_rfm.apply(kmeans_level, axis=1)

    loyal_kmeans = penjualan_rfm['RFM with KMeans Level'].value_counts()['Loyal']
    promising_kmeans = penjualan_rfm['RFM with KMeans Level'].value_counts()['Promising']
    need_attention_kmeans = penjualan_rfm['RFM with KMeans Level'].value_counts()['Need Attention']

    #Data RFM (Recency, Frequency, Monetary) With KMeans Clustering
    st.subheader("Data RFM (Recency, Frequency, Monetary) With KMeans Clustering")
    st.caption(f"Total Customer: {penjualan_rfm['Customer ID'].count()}")

    col1,col2,col3=st.columns(3)
    with col1:
            st.write("Loyal: ", loyal_kmeans)
    with col2:
            st.write("Promising: ", promising_kmeans)
    with col3:
            st.write("Need attention: ", need_attention_kmeans)

    st.markdown("""---""")

    col1, col2 = st.columns(2)
    with col1:
        k_means = KMeans(n_clusters=3, random_state=42)
        k_means.fit(data)
        labels = k_means.labels_
        penjualan_rfm['ClusterLabels'] = labels
        x_val = 'Recency'
        y_val = 'Frequency'
        z_val = 'Monetary'

        fig = px.scatter_3d(penjualan_rfm, x=x_val, y=y_val, z=z_val, color='ClusterLabels', labels='ClusterLabels', title="<b> RFM KMeans Clustering Customer Distribution</b>", color_continuous_scale='cividis')
        st.plotly_chart(fig,use_container_width=True,height=500)
        st.write("Yellow : Loyal")
        st.write("Blue  : Promising")
        st.write("Gray : Need Attention")

    with col2:
        st.subheader("RFM")
        #dataRFM=pd.melt(penjualan_rfm,value_vars=["Recency", "Frequency", "Monetary"], value_name="value")
        fig = px.pie(penjualan_rfm, values=[loyal_kmeans, promising_kmeans, need_attention_kmeans], names=["Loyal", "Promising", "Need Attention"])
        fig.update_traces(textposition = "inside")
        st.plotly_chart(fig,use_container_width=True)


    #movie_select_pie_format = pd.melt(movie_select_pie_format, id_vars=["presenter", "movie_name", "average_vote", "genre_name"], value_vars=["amount_wertvoll_1","amount_gut_2" ,"amount_okay_3", "amount_schlecht_4", "amount_grottig_5"], var_name="vote" ,value_name="value")


    st.markdown("""---""")




    #Create for Region
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    st.sidebar.header("Choose your Filter: ")
    region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())

    if not region:
        df2 =df.copy()
    else:
        df2 = df[df["Region"].isin(region)]

    #Create for Country
    country =st.sidebar.multiselect("Pick your Country", df2["Country"].unique())
    if not country:
        df3=df2.copy()
    else:
        df3 = df[df["Country"].isin(country)]

    #Create for State
    state =st.sidebar.multiselect("Pick your State", df2["State"].unique())

    # Filter Data based on Region, Country, n City

    if not region and not country and not state:
        filtered_df =df
    elif not state and not country:
        filtered_df = df[df["Region"].isin(region)]
    elif not region and not state:
        filtered_df=df[df["Country"].isin(country)]
    elif country and state:
        filtered_df=df3[df["Country"].isin(country) & df3["State"].isin(state)]
    elif region and state:
        filtered_df=df3[df["Region"].isin(region) & df3["State"].isin(state)]
    elif region and country:
        filtered_df=df3[df["Region"].isin(region) & df3["Country"].isin(country)]
    elif country:
        filtered_df = df[df3["Country"].isin(country)]
    else:
        filtered_df = df3[df3["Region"].isin(region)&df3["Country"].isin(country)&df3["State"].isin(state)]

    category_df = filtered_df.groupby(by = ["Category"], as_index = False)["Sales"].sum()

    cl1,cl2= st.columns(2)
    with col1:
        st.subheader("Category wise Sales")
        fig = px.bar(category_df, x = "Category", y="Sales", text=['${:,.2f}'.format(x) for x in category_df["Sales"]], template="seaborn")
        st.plotly_chart(fig,use_container_width=True, height=200)

    with col2:
        st.subheader("Region wise Sales")
        fig = px.pie(filtered_df, values ="Sales", names = "Region", hole=0.5)
        fig.update_traces(text =filtered_df["Region"], textposition = "outside")
        st.plotly_chart(fig,use_container_width=True)

    pie1,pie2=st.columns(2)
    with pie1:
        st.subheader('Segment wise Sales')
        fig =px.pie(filtered_df, values="Sales",names="Segment",template="plotly_dark")
        fig.update_traces(text=filtered_df["Segment"],textposition="inside")
        st.plotly_chart(fig,use_container_width=True)
    with pie2:
        st.subheader('Category wise Sales')
        fig =px.pie(filtered_df, values="Sales",names="Category",template="gridon")
        fig.update_traces(text=filtered_df["Category"],textposition="inside")
        st.plotly_chart(fig,use_container_width=True)

    cl1,cl2= st.columns(2)
    with cl1:
        with st.expander("Category_ViewData"):
            st.write(category_df.style.background_gradient(None))
            csv = category_df.to_csv(index = False).encode('utf-8')
        
    with cl2:
        with st.expander("Region_ViewData"):
            region=filtered_df.groupby(by="Region", as_index= False)["Sales"].sum()
            st.write(region.style.background_gradient(None))
            csv = region.to_csv(index = False).encode('utf-8')

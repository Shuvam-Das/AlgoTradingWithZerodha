"""Simple Streamlit UI to display order history and status."""
import streamlit as st
import pandas as pd
import sqlite3
import os

DB_PATH = '.orders.db'


def load_orders(db_path=DB_PATH):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM orders ORDER BY created_at DESC', conn)
    conn.close()
    return df


def main():
    st.title('AlgoTradingWithZerodha - Order History')
    df = load_orders()
    if df.empty:
        st.info('No orders found')
        return
    st.dataframe(df)


if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns={'Time': 'Datetime', 'value': 'Kwh'})
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

def main():
    st.title('Energy Data Analysis Dashboard')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Line chart
        st.subheader('Energy Consumption Over Time')
        st.line_chart(df.set_index('Datetime')['Kwh'])
        
        # Statistics
        st.subheader('Data Statistics')
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average", f"{df['Kwh'].mean():.2f}")
        with col2:
            st.metric("Maximum", f"{df['Kwh'].max():.2f}")
        with col3:
            st.metric("Minimum", f"{df['Kwh'].min():.2f}")
        with col4:
            st.metric("Total Records", len(df))
        
        # Raw data
        st.subheader('Raw Data')
        st.dataframe(df)

if __name__ == "__main__":
    main()

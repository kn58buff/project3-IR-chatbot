import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
st.title("Vizualizations")

if "query_log" not in st.session_state or not st.session_state.query_log:
    st.info("No queries logged yet.")
else:
    logs = pd.DataFrame(st.session_state.query_log)
    st.dataframe(logs)
    logs = logs.loc[logs["intent"] == "retrieval"]

    if logs.empty:
        st.info("No retrieval queries logged yet.")
    else:
        st.write(f"Total retrieval queries logged: {len(logs)}")
        all_topics = list(set(logs["selected_topics"].explode().unique().tolist()) | set(logs["detected_topics"].explode().unique().tolist()))

        st.subheader("Selected Topics")
        topics_exploded = logs.explode("selected_topics")
        topic_counts = topics_exploded["selected_topics"].value_counts().reset_index()
        topic_counts.columns = ["topic", "count_selected"]

        chart = alt.Chart(topic_counts).mark_bar().encode(
            x="topic",
            y="count_selected",
            tooltip=["topic", "count_selected"]
        )

        st.altair_chart(chart, width="stretch")

        st.subheader("Topics per query")
        chart = alt.Chart(topics_exploded).mark_bar().encode(
            x="query:N",
            y="count()",
            color="selected_topics:N",
            tooltip=["query", "selected_topics"]
        )
        st.altair_chart(chart, width="stretch")

        st.subheader("Detected Topics vs Selected Topics")
        detected_counts = logs.explode("detected_topics")["detected_topics"].value_counts().reset_index()
        detected_counts.columns = ["detected_topic", "count_detected"]
        

        st.dataframe(detected_counts.merge(topic_counts, left_on = "detected_topic", right_on = "topic", how = "outer").fillna(0))

        matrix = []
        for log in st.session_state.query_log:
            if log["intent"] != "retrieval":
                continue

            row = []
            detected = set(log["detected_topics"])
            selected = set(log["selected_topics"])

            for topic in all_topics:
                if topic in detected and topic in selected:
                    row.append(3)
                elif topic in detected:
                    row.append(1)
                elif topic in selected:
                    row.append(2)
                else:
                    row.append(0)
            matrix.append(row)
        matrix_df = pd.DataFrame(matrix, columns=all_topics)

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix_df, aspect = 'auto', cmap = 'Blues')
        ax.set_xticks(range(len(all_topics)))
        ax.set_xticklabels(all_topics, rotation=45, ha='right')
        ax.set_yticks(range(len(matrix_df)))
        ax.set_yticklabels([log["query"] for log in st.session_state.query_log if log["intent"] == "retrieval"])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Value')

        st.write("### Topic Selection Matrix")

        st.pyplot(fig)
        
        st.write("Legend: 0 - Detected & Not Selected, 1 - Detected Only, 2 - Selected Only, 3 - Detected & Selected")

        st.subheader("Query Logs")
        for log in st.session_state.query_log:
            if log["intent"] != "retrieval":
                continue
            st.write(f"### Query: {log['query']}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Detected Topics**")
                st.write(log["detected_topics"])
            with col2:
                st.write("**Selected Topics**")
                st.write(log["selected_topics"])
        st.subheader("Processing times")

        st.scatter_chart(logs["processing_time"])

        st.subheader("Query Length vs Processing Time")

        scatter_df = logs.copy()
        scatter_df["query_length"] = scatter_df["query"].str.split().str.len()

        ql_vs_pt_chart = alt.Chart(scatter_df).mark_circle(size=50).encode(
            x="query_length",
            y="processing_time",
            tooltip=["query", "processing_time"]
        )
        st.altair_chart(ql_vs_pt_chart, width="stretch")

        st.subheader("Query Log Data")
        logs["query_length"] = logs["query"].apply(lambda x: len(x.split()))
        logs["response_length"] = logs["bot_response"].apply(lambda x: len(x.split()) if x else 0)

        st.metric("Average Query Length", f"{logs['query_length'].mean():.2f} words")
        st.metric("Average Response Length", f"{logs['response_length'].mean():.2f} words")

        st.subheader("Response Length Distribution")
        st.bar_chart(logs["response_length"])

        st.subheader("Response Length vs Relevance Score")
        if "relevance_scores" in logs.columns:
            resp_df = logs.explode("relevance_scores")
            resp_df["relevance_scores"] = resp_df["relevance_scores"].astype(float)

            chart = alt.Chart(resp_df).mark_circle().encode(
                x="response_length",
                y="relevance_scores",
                tooltip=["query", "response_length", "relevance_scores"]
            )
            st.altair_chart(chart, width="stretch")

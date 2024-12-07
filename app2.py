import streamlit as st
from youtube import *
from query_resolver import qa, fine_tuned_model
from de import generateQuestions
from youtube import extract_topics, search_youtube_for_topics, cluster_topics_and_videos
from PIL import Image
import requests
from io import BytesIO


def main():
    st.title("📚 Academic Helper Suite")
    st.markdown("""
    Welcome to the **Academic Helper Suite**! This application has three main features:
    1. **Contextual Query Resolver**: Ask follow-up questions based on a given context.
    2. **YouTube Video Recommender**: Get curated video recommendations for academic topics.
    3. **Enhanced MCQ Generator**: Generate multiple-choice questions (MCQs) from a given text.
    """)

    # Tabbed interface for better organization
    tabs = st.tabs(["📖 Query Resolver", "🎥 Video Recommender", "📝 MCQ Generator"])

    # Contextual Query Resolver
    with tabs[0]:
        st.header("📖 Contextual Query Resolver")
        st.markdown("""
            **Instructions**:
            - Enter a **context** to set the background information.
            - Ask **follow-up questions** based on the context.
            - Use the **Reset Context** button to clear the current context.
        """)

        # Session state for context and Q&A history
        if "context" not in st.session_state:
            st.session_state.context = ""
        if "questions" not in st.session_state:
            st.session_state.questions = []

        # Context input
        context_input = st.text_area("Enter context:", value=st.session_state.context, height=200)
        if st.button("Set Context"):
            st.session_state.context = context_input
            st.session_state.questions = []
            st.success("Context updated!")

        if st.button("Reset Context"):
            st.session_state.context = ""
            st.session_state.questions = []
            st.info("Context cleared!")

        # Display the current context
        if st.session_state.context:
            st.subheader("📖 Current Context")
            st.write(st.session_state.context)

            # Question input
            question = st.text_input("Enter your question:")
            if st.button("Ask Question"):
                if question.strip():
                    answer = qa(fine_tuned_model, st.session_state.context, question)
                    st.session_state.questions.append({"question": question, "answer": answer})
                    st.success("Answer generated!")
                else:
                    st.error("Please enter a valid question.")

            # Display Q&A history in a collapsible section
            if st.session_state.questions:
                with st.expander("📝 View Question & Answer History"):
                    for idx, qa_pair in enumerate(st.session_state.questions):
                        st.markdown(f"**Q{idx + 1}:** {qa_pair['question']}")
                        for sent in qa_pair['answer']:
                            st.write(f"  - {sent}")

        else:
            st.warning("No context set. Please enter one to proceed.")

    # YouTube Video Recommender
    with tabs[1]:
        st.header("🎥 YouTube Video Recommender")
        st.markdown("Enhance your learning with curated video recommendations!")

        input_text = st.text_area("📝 Enter your academic text here:", height=150)
        if st.button("Recommend Videos"):
            if input_text.strip():
                st.info("Extracting topics and finding relevant videos...")
                
                # Extract topics from input text
                topics = extract_topics(input_text, n_topics=5)
                
                # Search for YouTube videos based on extracted topics
                video_suggestions = search_youtube_for_topics(topics)
                
                # Cluster the topics and videos
                clustered_videos = cluster_topics_and_videos(topics, video_suggestions)

                # Display videos in collapsible sections
                for cluster, videos in clustered_videos.items():
                    with st.expander(f"📚 Topic: {cluster}"):
                        for video in videos:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                thumbnail_url = video['thumbnail']
                                if isinstance(thumbnail_url, dict):
                                # Select one of the URLs; here we choose 'static' or any preferred key
                                    thumbnail_url = thumbnail_url.get('static') or thumbnail_url.get('rich')
                                try:
                                    # Check if the thumbnail URL is valid
                                    response = requests.get(thumbnail_url)
                                    if response.status_code == 200:
                                        image = Image.open(BytesIO(response.content))
                                        st.image(image, width=150)
                                    else:
                                        st.warning("Thumbnail not available.")
                                except Exception as e:
                                    st.warning(f"Error loading thumbnail: {e}")
                            
                            with col2:
                                st.markdown(f"**🎬 {video['title']}**")
                                st.write(video['description'])
                                st.markdown(f"[🔗 Watch Video]({video['url']})")
            else:
                st.error("⚠️ Please enter some text!")
    # Enhanced MCQ Generator
    with tabs[2]:
        st.header("📝 Enhanced MCQ Generator")
        st.markdown("""
            Enter a passage of text, and this tool will generate multiple-choice questions (MCQs). 
            Click "Show Answer" to reveal the correct option.
        """)

        text = st.text_area("Enter your text:", height=200)
        count = st.number_input("Number of MCQs to generate:", min_value=1, max_value=10, value=5)

        if st.button("Generate MCQs"):
            if not text.strip():
                st.error("Please enter valid text!")
            else:
                with st.spinner("Generating questions..."):
                    try:
                        questions = generateQuestions(text, count)

                        if not questions:
                            st.warning("No questions generated. Check your input or try again.")
                            st.write("Generated Questions:", questions)
                        else:
                            for idx, mcq in enumerate(questions):
                                st.markdown(f"### Question {idx + 1}")
                                st.write(mcq["question"])

                                # Display options
                                for option in mcq["options"]:
                                    st.write(f"- {option}")

                                with st.expander("Show Answer"):
                                    st.success(f"Correct Answer: {mcq['answer']}")
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

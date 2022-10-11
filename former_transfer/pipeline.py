import configparser
import glob
import sys

import article_selection.article_selection as article_selection
from sentiment_analysis.inference import calulate_sentiment, eval_sentiment
from sentiment_analysis.word2vec_sentiment import *
from visualization.dash_plot import dash_plot
from visualization.wordcloud import generate_word_clouds

if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config = configparser.ConfigParser()
        config.read(config_file)
    except IndexError:
        quit()

    if config.getboolean("ArticleSelection", "run_article_selection"):
        base_path = config.get("ArticleSelection", "input_path_base")
        start_year = config.getint("ArticleSelection", "start_year")
        end_year = config.getint("ArticleSelection", "end_year")
        data_path_list = [base_path + str(year) + "/" for year in range(start_year, end_year + 1)]
        json_file_list = []
        for path in data_path_list:
            json_file_list += [file_path for file_path in glob.glob(path + "*.json")]

        search_keywords = config.get("ArticleSelection", "search_words").lower().split(", ")
        output_base = config.get("ArticleSelection", "output_base")

        create_new_files = not config.getboolean("ArticleSelection", "append_to_existing_file")

        use_annotation = config.getboolean("ArticleSelection", "use_annotation")
        training_size = config.getint("ArticleSelection", "training_size")
        seed = config.getint("ArticleSelection", "seed")

        article_selection.write_relevant_content_to_file(json_file_list,
                                                         output_base,
                                                         search_keywords=search_keywords,
                                                         new=create_new_files,
                                                         training_size=training_size,
                                                         seed=seed,
                                                         annotation=use_annotation)

    if config.getboolean("Analysis", "run_w2v"):
        input_file = config.get("Analysis", "input_file")
        search_words = config.get("Analysis", "search_words_w2v").lower().split(",")
        base_output_path = config.get("Analysis", "output_base_w2v")
        start_year = config.getint("Analysis", "start_year")
        end_year = config.getint("Analysis", "end_year")
        number_most_sim = config.getint("Analysis", "number_most_sim")

        if config.getboolean("Analysis", "run_by_year"):
            similarity_by_year(input_file, base_output_path, search_words,
                               start_year, end_year, number_most_sim)

        if config.getboolean("Analysis", "run_by_publisher"):
            similarity_by_publisher(input_file, base_output_path, search_words,
                                    start_year, end_year, number_most_sim)

        if config.getboolean("Analysis", "run_by_publisher_by_year"):
            similarity_by_year_and_publisher(input_file, base_output_path,
                                             search_words, start_year,
                                             end_year, number_most_sim)

    if config.getboolean("Analysis", "run_senti"):
        input_file = config.get("Analysis", "input_file")
        search_words = config.get("Analysis", "search_words").lower().split(",")
        output_file = config.get("Analysis", "output_senti")
        methods = config.get('Analysis', 'senti_methods').lower().split(", ")
        finetuned_sentibert_path = config.get('Analysis', 'finetuned_sentibert_path')

        calulate_sentiment(
            input_file,
            output_file,
            search_words,
            methods=methods,
            finetuned_sentibert_path=finetuned_sentibert_path
        )


    if config.getboolean("Analysis", "run_senti_eval"):
        senti_eval_input = config.get("Analysis", "senti_eval_input")
        search_words = config.get("Analysis", "search_words").lower().split(",")
        senti_eval_output = config.get("Analysis", "senti_eval_output")
        methods = config.get('Analysis', 'senti_methods').lower().split(", ")
        finetuned_sentibert_path = config.get('Analysis', 'finetuned_sentibert_path')

        eval_sentiment(
            senti_eval_input,
            senti_eval_output,
            search_words,
            methods=methods,
            finetuned_sentibert_path=finetuned_sentibert_path
        )

    if config.getboolean("Plotting", "sentiment_plot"):
        input_file = config.get("Plotting", "input_file")
        dash_plot(input_file)

    if config.getboolean("WordClouds", "wordcloud_plot"):
        input_file = config.get("WordClouds", "input_file")
        output_path = config.get("WordClouds", "output_path")
        words = config.get("WordClouds", "words").lower().split(", ")
        column_values = config.get("WordClouds", "column_values").lower().split(", ")
        number_of_words_in_wordcloud = config.getint("WordClouds", "number_of_words_in_wordcloud")

        generate_word_clouds(input_file, words, column_values, output_path, number_of_words_in_wordcloud)
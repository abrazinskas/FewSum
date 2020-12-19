# Abstractive Summaries for Amazon and Yelp

This folder contains gold summaries that were used for fine-tuning and evaluation.

For Amazon, we re-used reviews and formally written summaries from [Copycat](https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer), 15 products per each category. The categories are listed below.

* Electronics
* Clothing, Shoes and Jewelry
* Home and Kitchen
* Personal Care


For Yelp, we sampled 100 businesses.


## Example Summaries

> This laptop sleeve is good quality and it fits a 14 inch laptop nicely. It can have a strong rubber smell initially, but with time the smell should fade. The sleeve provides great protection from scratches and looks great. It is a good value and is recommended for laptop protection.

> The bagels are really good, and the price isn't too expensive. The staff is generally friendly, helpful and professional, although the wait time can be as high as 15 to 20 minutes.

> These shoes run true to size, do a good job supporting the arch of the foot and are well-suited for exercise. They're good looking, comfortable, and the sole feels soft and cushioned. Overall they are a nice, light-weight pair of shoes and come in a variety of stylish colors.

## Text Statistics

| Dataset | Products/businesses  | Summaries | Avg. word count | Avg. sentence count |
| :---:  | :---:  | :---:  | :---:  | :---:  |
| Amazon  | 60  | 180 | 56.92 | 3.68 |
| Yelp  | 100  | 300 | 58.06 | 4.30 |

## Data Annotation Process

We used the Amazon Mechanical Turk platform by hiring 3 workers per data-instance. They were presented with 8 reviews and asked to write a summary.

## Annotator Requirements

To obtain high quality annotations, we set high bar annotator requirements. Additionally, we used a qualification test to make sure that our instructions are well understood.

* Are native English speakers.
* Have at least 1000 completed Amazon tasks (HITs).
* Have 98% approval rate.
* Located in the USA, UK, or Canada.
* Passed a qualification test that assured their understand of the task.


## Annotation Instructions

The workers were asked to read the reviews and by following the guidelines below to write a summary in their own words.

* The summary should reflect user common opinions expressed in the reviews. Try to preserve the common sentiment of the opinions and their details (e.g. what exactly the users like or dislike). For example, if most reviews are negative about the sound quality, then also write negatively about it.
* Please make the summary coherent and fluent in terms of sentence and information structure. Iterate over the written summary multiple times to improve it, and re-read the reviews whenever necessary.
* The summary should not look like a review, please write formally.
* Keep the length of the summary reasonably close to the average length of the reviews.
* Please try to write the summary using your own words instead of copying text directly from the reviews. Using the exact words from the reviews is allowed but do not copy more than 5 consecutive words from a review.


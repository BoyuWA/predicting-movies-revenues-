---
title: 'Final project'
author: 'Boyu Wu'
date: 'Due: December 19 2021'
output:
  pdf_document:
    toc: no
    toc_depth: '4'
  html_document:
    highlight: haddock
    number_sections: yes
    theme: lumen
    toc: no
    toc_depth: 4
    toc_float: yes
  bookdown::pdf_document2:
    number_sections: yes
    toc: yes
    toc_depth: '3'
urlcolor: blue
---
---

```{r setup, include=FALSE}
options(scipen = 0, digits = 3)  # controls number of significant digits printed
knitr::opts_chunk$set(warning = FALSE, message = FALSE) 
```


```{r, message = FALSE,echo=FALSE}
library(rpart)             # install.packages("rpart")
library(rpart.plot)        # install.packages("rpart.plot")
library(tidyverse)
library(Metrics)
library(randomForest)  # random forests
library(kableExtra)    # for printing tables
library(psych)
```
# Executive Summary:

**Problem**
In this final project, we want to analyze the effects of data on the lifetime gross of a movie. 
**Data**
We will try to predict what metadata influences the money-making aspect the most.
This data we extracted from multiple sources:

[main data - movielens](https://grouplens.org/datasets/movielens/)                      
[imbd data](https://www.imdb.com/interfaces/)                               
[boxoffice](https://data.world/eliasdabbas/boxofficemojo-alltime-domestic-data)

**Analysis**
Especially we will do use linear regression methods and trees together with ensemble methods to see how well we can predict the response. And look whether a prediction of this kind is even possible. 
Our analysis of this is being conducted as a regression problem. So that if someone wants they can also use our code and predict the gross. 
**Conclusion**
In the end, we managed to tune an ensemble method that predicts almost 50% of the variance, which can be seen as quite good in terms of prediction. Here is our [github repo](https://github.com/BoyuWA/predicting-movies-revenues-)

# Introduction
**Background information**
The context of this project is to run a regression program that can predict the gross of a movie without seeing it. So just by looking at movie review averages, how many people revised it, runtime,we want to predict the earning. This way we can help the movie industry to gain a first impression of how well the movie will perform based on metadata. Through this process, we can also help them to reduce the time that they need to see and damage control stuff in the real world, but they can directly release the movie being sure that it will perform or not perform according to the predictions.
**Analysis goal**
The goal is to obtain a good model that can accurately depict the dependencies of metadata and lifetime gross. But this model should also be easily explainable and logical. Thus, we opted against doing a nural network.
We will not only use features such as runtime, ratings, release year, but we will also onehot encode different genres. As we know there are some genres that generally perform better than others. Success will be evaluated based on either r squared or RSME. 
**Significance**
We need to predict the dependencies accurately and then we only then can put our product (the code) on the market and help the movie industry with their new film releases. 

**we decided to download the data instead of downloading from a url because some files were extremely large and it took too much time to do it every time from the internet.**
**some of the files can be found in the github files but some are too large to upload them.**

```{r}
#due to te fact that movie data is split among many different
#files we need to read in the individual files
set.seed(11)

movie_raw = read.csv("movies.csv")
rating_raw_grouplense = read.csv("ratings.csv")
rating_raw_imbd = read_tsv("data.tsv",show_col_types = FALSE)
links = read.csv("links.csv")
boxoffice = read.csv("boxoffice.csv")
runtime = read_tsv("runtime.tsv",show_col_types = FALSE)
title = read_tsv("title.tsv",show_col_types = FALSE)


#transfor it into the datastructure we are familiar with

movie_raw = movie_raw %>% as_tibble 
rating_raw_grouplense = rating_raw_grouplense %>% as_tibble 
rating_raw_imbd = rating_raw_imbd %>% as_tibble 
links = links %>% as_tibble 
boxoffice = boxoffice %>% as_tibble
runtime = runtime %>% as_tibble
title = title %>% as_tibble
```


```{r,echo=FALSE}
title = title%>%
  select(c('titleId','title','isOriginalTitle'))
```



**Links: Identifiers that can be used to link to other sources of movie data are contained in the file links.csv. Each line of this file after the header row represents one movie. Allows us to merge our main data with data (in our case) from imdb**.
**One imporatnt factor is we need to mutate the link so that we can join the data in the end because our imdbId requires a tt followed by a 7 digit number, we need to tidy our data so that this is possilbe.**

```{r}
links %>% arrange(imdbId) #print links tibble
links$imdbId = sprintf("%07d",links$imdbId) 
links = links %>% 
  mutate(imdbId = paste("tt", 
                        imdbId, 
                        collapse = NULL, 
                        sep = ""))

links
```
**Runtime: this is one of our bigger datasets with a few interesting things we want to extract. Among others, tconst for merging and isAdult and runtimeMinutes for data analysis (we want to use as many features we can get our hands on as possible (ofcourse these features need to have an influence over our predicted boxoffice results).**

```{r,error=FALSE}
runtime = runtime %>%
    mutate_at(
      vars(tconst, isAdult,runtimeMinutes),
      na_if, "\\N") %>%
  select(
    c("tconst","isAdult","runtimeMinutes")) %>%
  drop_na()
runtime
```

**Boxoffice: Lifetime_gross is included in the boxoffice.csv. This data represents the best >15k grossing films all time till 2019 - originally this data is parsed from Boxofficemojo.**

```{r}
boxoffice = boxoffice %>%                   #filter boxoffice data
  mutate(wtitle = str_to_lower(title)) %>% 
  select(c("rank","title","lifetime_gross"))    
boxoffice                                   #print out boxoffice
```

**Movie_raw:  Movie information is contained in the file movies.csv. Each line of this file after the header row represents one movie. This is our main "frame". This data we still need to tidy up:**
 **-we have release year and title in one and we need to one hot encode genres.**

```{r}
movie_raw
movie_raw = movie_raw %>% 
  separate(title,                                    #seperate release year 
           into = c("title","year"),                 #and movie title
           sep = "\\(" ) %>%
  separate(year,into = c("year", "xxx"), sep = "\\)") %>%
  select(-xxx) %>% 
  filter(str_detect(year,"^[1-9][0-9][0-9][0-9]"))%>%#regex pattern matching
  mutate(year = as.integer(year))
movie_raw = movie_raw %>% mutate(year = as.integer(year))

genres_df = movie_raw %>% mutate(value = 1) %>%      #one hot encoding for genre
  separate_rows(genres, sep = "\\|") %>%
  spread(genres, value, fill =0)


genres_df = genres_df %>%                            #mutate a lowercase title
  mutate(wtitle = str_to_lower(title))

boxoffice = boxoffice %>%                            #mutate a lowercase title
  mutate(wtitle = str_to_lower(title))

genres_df
```
```{r}
genres_df
genres_df_big = genres_df %>% 
  inner_join(links, by = c("movieId"))             #testing our inner join
```


**We realize that something is wrong! We obviously have way too little data in the merged tibble. One possibility was that it has something to do with lower and uppercase, but after implementing lowercase for all, this still hasnt fixed the problem.**
**If we look into the our genre data we will realize that we can only extract the whole row if we add a space after the last character. Thus we removed the last space to check again. -> luckily for us this solved the merging problem.**
```{r}
genres_df
#demonstration code
genres_df %>% filter(title == "Toy Story") 
genres_df %>% filter(title == "Toy Story ")
```

```{r}
genres_df
boxoffice
genres_df = genres_df %>%
  mutate(wtitle = across(where(is.character), 
                         str_remove_all, 
                         pattern = fixed(" "))[[2]]) #uniformly cleaning of data
boxoffice = boxoffice %>%
  mutate(wtitle = across(where(is.character), 
                         str_remove_all, 
                         pattern = fixed(" "))[[2]])%>%                            #mutate a lowercase title
  mutate(wtitle = str_to_lower(wtitle)) #uniformly cleaning of data
boxoffice = genres_df %>% select(-title) %>% inner_join(boxoffice, by = c("wtitle")) 
```

**Rating_grouplense: All ratings are contained in the file ratings.csv. Each line of this file after the header row represents one rating of one movie by one user. This is one of the ratings we will make use of.**

```{r}
rating_grouplense = rating_raw_grouplense %>% select(c("movieId","rating")) %>% group_by(movieId) %>% summarise(avg_rating_grouplense = mean(rating))
rating_grouplense
```


**Here we join the raw imbd data with our runtime data.**

```{r}
rating_raw_imbd

imbd_data = rating_raw_imbd %>% left_join(runtime, by = c("tconst"))



#movie_raw %>% left_join(data_2, by = c("column_1","column_2"))
```
**Now we will arrange the data by numVotes to verify our plausabilty of the data.**
```{r}
imbd_data = imbd_data %>% filter(!isAdult>1) %>% arrange(desc(numVotes))

imbd_data
```
```{r}
boxoffice 
title = title %>%
  mutate(wtitle = across(where(is.character), 
                         str_remove_all, 
                         pattern = fixed(" "))[[2]])%>%
  mutate(wtitle = str_to_lower(wtitle)) %>% select(-title) %>% filter(isOriginalTitle == 1)
title = title %>% filter(isOriginalTitle == 1)


merge1 = boxoffice %>%  #merge1 is the merge with our titles and our boxoffice results
  left_join(title, by = c("wtitle")) %>% distinct()
merge1
```

**We are now realizing a small problem lets take black panther there are 2 same movies that are flagged as originals, this is the due to the fact that one of them was the 1996 version and the other one the 2018 movie that some of us watched in theaters.**
**This problem is a one we cant easily fix but well try in the follwing to minimize the effect by dropping as many duplicates as possible**

```{r}
imbd_data = imbd_data %>% mutate(titleId = tconst) %>% select(-tconst)
imbd_data
merge2 = merge1 %>% inner_join(imbd_data, by = c("titleId")) #first merge to add avarageRating (imdb numVotes isAdult and runtimeMinutes to it)
links = links %>% mutate(titleId = imdbId) %>% select(-imdbId)
merge2
merge3 = merge2 %>% select(-movieId) %>% inner_join(links, by = c("titleId")) #adding our link data into it this enables us to add the other rating database
merge3
merge4 = merge3 %>% left_join(rating_grouplense, by = c("movieId")) #we now add grouplense rating into it 

merge4
main_df = merge4 %>% relocate(lifetime_gross) %>% 
  select(-c("wtitle","rank","isOriginalTitle","titleId","movieId","tmdbId","title")) %>%  mutate(runtimeMinutes = as.integer(runtimeMinutes)) #we shall remove rank caus it highly correlates with revenue thus is a useless indicator
main_df 
#rating_grouplense
```

**Description of the final data**

**We have 15,984 observations in the data and 27 features**

**Each observation represent a movie**

**lifetime_gross -(continuous) this is our respone variable and its a numeric column that shows the total revenues of each film**

**year -(continuous) the year the movie was made**

**(no geners listed) -(categorical) if the movie didnt have any genere listed**

**columns 4-19 - (categorical) one hot encoding that each one represent a different genere**

**averageRating - (continuous) the average rating the movie got in IMBd**

**numVotes - (continuous) the number of votes for the rating**

**isAdult - (categorical) if the movie is for adults**

**runtimeMinutes - (continuous) the amount of time in minutes each movie is**

**avg_rating_grouplense - (continuous) the average rating the movie got in grouplense**



# EDA 
**After a very very long process of data tyding and wrangling we will do some exploratory data analysis**

**lets look at the data**
```{r}
describe(main_df)
```

**We can see a few intresting things from this.**
**For example we can see that the isadult column have a mean nedian and sd of 0 meaning the entire column is zero and therefor we should remove it.**
**Another thing is that we can see that the maximum revenue a movie made is 760 million dollars.**
```{r}
main_df = main_df%>%select(-isAdult)
```


**We will plot a few dependencies on lifetime_gross, to get a better overview of our data.**
```{r}
main_df %>% ggplot(aes(y = lifetime_gross, x = year)) +
  geom_point() + 
  geom_smooth()
```
**According to our data we can see there is a slight decrease of lifetime_gross over the duration of a movie. This makes sense, older movies in general have more time passed so that you could earn more money over a longer duration.**

\newpage

```{r}
main_df %>% ggplot(aes(y = lifetime_gross, x = averageRating)) +
  geom_point() + 
  geom_smooth()
```
**Interesting enough there seems to be a curve if we compare imdb ratings and gross.**
**Easy explainable is the right side where a higher ration correlates with more revenue but interesting enough bad movies also get a lot of revenue. Lets compare that to our second movie rating dataset!**

\newpage

```{r}
main_df %>% ggplot(aes(y = lifetime_gross, x = avg_rating_grouplense)) +
  geom_point() + 
  geom_smooth()
```
**On the other hand we see that movies with a rating of 4 in gouplense have higher lfetime gross tha other movies. One might explain this because people that rate movies watch alot of movies and thus some concept of the movies a critic might find overused while the casual watcher (here is where the company make the most money) enjoy it more.**

\newpage

```{r}
main_df %>% ggplot(aes(y = lifetime_gross, x = runtimeMinutes)) +
  geom_point() + 
  geom_smooth()
```
**And lastly it seems like that there is a perfect length of a movie which is around 120 minutes.**

\newpage


# Methods and Predictions

# Linear regression

**Now we shall run some analysis on our data.**

```{r}
set.seed(11)
n = nrow(main_df)
train_samples = sample(1:n, round(0.8*n))
```

```{r}
main_df_train = main_df[train_samples,]
main_df_test = main_df[-train_samples,]
```



```{r}
lm_fit = lm(lifetime_gross ~ avg_rating_grouplense, data = main_df_train)

prediction = predict(lm_fit,  newdata = main_df_test)

tibble("Predicted gross" = prediction,
       "Actual gross" = main_df_test$lifetime_gross) %>%
  head(10)
```

\newpage
```{r}
summary(lm_fit)

```
\newpage
```{r}
lm_fit = lm(lifetime_gross ~ averageRating, data = main_df_train)

prediction = predict(lm_fit,  newdata = main_df_test)

tibble("Predicted gross" = prediction,
       "Actual gross" = main_df_test$lifetime_gross) %>%
  head(10)
```
\newpage

```{r}
summary(lm_fit)


```

**we can see that a lm model is not very suitable for our data, this is because we are only looking at one variable so lets expand it and to have a linear regression for multiple variables **

\newpage
```{r}
lm_fit = lm(lifetime_gross ~., data = main_df_train)
summary(lm_fit)

rmse(main_df_test$lifetime_gross,prediction) 
```
**Now we can see that we got much better results.Our adjusted r-squared increased to 0.348.**
**Further more we can see a few interesting results.**
**year seems to have a negative impact on the revenue as we already saw in our EDA section.**
**Different generes seems to have different impact on the revenue like animation and action have a positive impact where drama and documentary have a negative one. This is to be expected since we know mosr blockbusters are action and animation where drama and documentary are less of a money grab.**

\newpage

# Decision Trees
```{r}
deep_tree_fit = rpart(lifetime_gross ~ .,
                         method = "anova",
                         parms = list(split = "gini"),
                         control = rpart.control(minsplit = 5),
                         data = main_df_train)
rpart.plot(deep_tree_fit) 
deep_tree_fit
```
**We can see that number of votes seems to be the leading factor in deciding the splits splits**
**This is a non prune tree so lets get a better tree and see the results**
\newpage
```{r}
cp_table = deep_tree_fit$cptable %>% as_tibble()

cp_table %>%
  filter(nsplit >= 2) %>%
  ggplot(aes(x = nsplit+1, y = xerror,
             ymin = xerror - xstd, ymax = xerror + xstd)) + 
  scale_x_log10() +
  geom_point() + geom_line() +
  geom_errorbar(width = 0.25) +
  xlab("Number of terminal nodes on log scale") + ylab("CV error") + 
  geom_hline(aes(yintercept = min(xerror)), linetype = "dashed") + 
  theme_bw()
```

**We can see from the plot above that the error keep decreasing which indicate that we do need a longer tree, the problem is that we don't really have a lot of observatons to do it becuase from tree looks most of the nods stoped due to lack f observation in them and so even if we didn't get overfitt yet we should just continue like thi. **

\newpage

```{r}
optimal_tree_info = cp_table %>% 
  filter(xerror - xstd < min(xerror)) %>% 
  arrange(nsplit) %>% head(1)
optimal_tree_info$nsplit

optimal_tree = prune(tree = deep_tree_fit, cp = optimal_tree_info$CP)
rpart.plot(optimal_tree) 
prediction = predict(optimal_tree, main_df_test) 

rmse(prediction, main_df_test$lifetime_gross) #52072302

```

**We see the pruned tree is a much smaller. Pruning successfull, the RMSE althogh its lower than the one we got in the linear regression its still quite high. We shall now use randomForest to reduce variance in our data. This will hopefully improve our predictive model.**

\newpage

**One problem with randomForest is that the names of the columns are not allowed to contain special characters else the randomForest command will not compile, thus we will mutate our column names before we run the model function,**
```{r}
a = main_df_train %>% mutate(nogenre = `(no genres listed)`, filmnoir = `Film-Noir`, scifi = `Sci-Fi`) %>% select(-c('(no genres listed)','Film-Noir','Sci-Fi'))

rf_fit = randomForest(lifetime_gross ~ ., data = a, mtry = 3,
                         importance = TRUE, na.action = na.omit)
num_features = ncol(a) - 1
mtry = floor(sqrt(num_features))
mtry
rf_fit
```
# Conclusion
**In conclusion we can see that our tested metrics do have a correlation with lifetime gross of a movie, we could also see that for example a few metrics are more influential then other. One such example we have showed in a broader way in our regression models where we compared the two ratings - imdb and grouplense. Here we could see that the imdb metric has a higher (linear) correlation with our response.**
**The best predictive methods are ensamble methods. All of these models come to a conclusion that we can predict the lifetime gross of a movie just by analysing meta data and without watching the movie itself.**


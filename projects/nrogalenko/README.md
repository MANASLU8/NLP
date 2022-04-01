**Студент:** Рогаленко Никита Александрович

Проект, являющийся результатом прохождения курса.

## Структура проекта

1. Директория `assets` содержит вспомогательные и конфигурационные файлы проекта (не являющиеся исходным кодом), в том числе - результаты обработки исходных датасетов. 
2. Файл `README.md` содержит общее описание проекта, инструкцию по запуску программных модулей и тестов для них. 
3. Директория `source` содержит файлы с исходным кодом проекта.

- Пакет `text_annotation` содержит инструменты для токенизации с последующей нормализацией токенов и выводом аннотации документа в отдельный tsv файл.
- Пакет `typos_correction` содержит модуль для исправления опечаток в тексте с использованием словаря, создаваемого на основе tsv-аннотации
- Пакет `text_vectorization` содержит инструменты для создания словаря наиболее частых токенов, матрицы термин-документ, а также векторизации текста на основе
метрики tf-idf. Также содержит модуль для обучения модели w2v и визуализации полученных с ее помощью векторов, сравнивая результаты векторизации с базовым методом.
- Пакет `topic_modelling` содержит инструменты для кластеризации обрабатываемого текста с использованием модели LDA с заданным количеством
тем и итераций. Также для каждого эксперимента в директории `assets/topic-modelling-results` сохраняются 10 самых частых слов и наиболее вероятных документов для каждой темы,
распределение вероятности принадлежности документа к каждой из выделенных тем, а также `perplexity`. Строится график зависимости данной метрики от количества выделенных тем. 
- Пакет `text_classification` содержит инструменты для классификации текстов с использованием модели SVM с различными ядерными функциями. Для каждого эксперимента фиксируется
набор метрик, отражающих эффективность модели. Также дополнительно происходят эксперименты с сокращением размерности векторов для определения характера влияния
изменения размерности на метрики.
4. директория `source/tests` содержит модульные тесты

## Настройка окружения

При разработке использовался Python 3.10

Выбран датасет №2 - https://huggingface.co/datasets/ag_news. В связи с этим необходимо, 
чтобы в директории `assets` располагались файлы `train.csv` и `test.csv` с обрабатываемыми новостными текстами

При работе с модулем исправления опечаток необходимо, чтобы в директории
`assets` располагался файл с опечатками `test-corrupted.csv` (https://drive.google.com/drive/folders/1cE5m5EfhFGfcRx4dOJeqNksynPOafYwr файл ag-news-test-corrupted.tar.xz)

## Запуск проекта

Для запуска необходимо выполнить следующее:

```
cd projects/nrogalenko/source
python __main__.py
```

### Лабораторная №1

Результатом выполнения является набор набор аннотаций со следующей структурой:

`<sentence_X_token_N>   <sentence_X_stem_N>    <sentence_X_lemma_N>    <sentence_X_pos_tag_N>    <sentence_X_token_tag_N>`

Где pos_tag - POS tag, обозначающий часть речи. А token_tag - класс токена, выделенный в рамках работы

### Лабораторная №2

Результатом выполнения являются числа, харакетризующие эффективность работы модуля исправления опечаток.

```
correct documents tokens sum number 12916
correct documents tokens in corrupted before module 10575
correct documents tokens in corrupted after module 12208
Before module: 0.818751935583772 After module: 0.9451842675751007
```
### Лабораторная №3

Результатом выполнения являются графики, отражающие расстояние тестовых токенов друг относительно друга, при использовании разных способов викторизации, а также косинусное расстояние между словами.
Также формируется файл `test-embeddings.tsv`, содержащий векторные представления документов тестовой выборки, полученных
согласно алгоритму, описанному в задании.

### Лабораторная №4

Сначала из словаря и аннотированных документов составляется csr-матрица термин-документ, на основе которой происходит обучение модели
с заданными значениями. В качестве результата в директории `assets/topic-modelling-results` сохраняются 10 самых частых слов и наиболее вероятных документов для каждой темы,
распределение вероятности принадлежности документа к каждой из выделенных тем, а также `perplexity`. Также строится график зависимости метрики от количества тем. Пример файла:
```
Topic 1:
win ap game new first team one season year last
"2","Patriot League capsules","DUXBURY Coach: Don Dellorco (22d year, 143-72-9). Last year's record: 9-1-2. Returning starters (6): Peter Bizinkauskas, TE, 6-1, 190, Sr.; Tim Confer, TB/CB, 6-0, 180, Sr.; Shaun Croscup, C, 6-0, 225, Sr.; Matt Johnston, NG, 5-10, 180, Sr.; Ryan Mullin, FB/LB, 5-8, 170, Jr.; Chris Nixon, TB, 5-10, 170, Jr. Returning lettermen (11): Steve Ahern, S, Sr.; Tim Griswold, OT/DT, ..."
"2","Greater Boston League capsules","ARLINGTON Coach: Rob DiLoreto (third year, 9-11). Last year's record: 4-6. Returning starters (12): Jay McGrath, C, 6-0, 260, Sr.; Michael O'Loughlin, RB, 5-11, 185, Sr.; Moses Ortiz, TE, 6-1 215, Sr.; Jordan Cooper, SE, 6-3, 195, Sr.; Neil Rainford, RB, 5-10, 210, Sr.; Michael Talarico, RB, 6-0, 195, Jr.; Peter Samko, OG, 5-9, 255, Jr.; Josh Vest, DL, 6-0, ..."
"2","Northeastern Conference","BEVERLY Coach: Dan Bauer (second year). Last year's record: 4-7. Returning starters (7): Nate Boynton, OT/DT, 6-3, 290, Sr.; Armando Cuko, kicker/punter, 5-11, 180, Sr.; Justin Fisher, OL/DL, 5-9, 230, Sr.; Dan Abate, WR/DB, 5-8, 165, Sr.; Travis Anderson, WR/DB, 5-8, 175, Sr.; Chris Kruczynski, OL/DL, 6-1, 240, Sr.; Jason Comeau, C/DL, 6-0, 215, Sr. Returning lettermen: Nick Abraham, RB/LB, ..."
"2","2005 Red Sox schedule","(Home games in caps) APRIL 4 Mon.at Yankees 1:05 6 Wed.at Yankees 7:05 7 Thu.at Yankees 7:05 8 Fri.at Toronto TBD 9 Sat.at Toronto 1:05 10 Sun.at Toronto 1:05 11 Mon.YANKEES 3:05 13 Wed.YANKEES 7:05 14 Thu.YANKEES 7:05 15 Fri.TAMPA BAY 7:05 16 Sat.TAMPA BAY TBD 17 Sun.TAMPA BAY 2:05 18 Mon.TORONTO 11:05 19 Tue.TORONTO 7:05 20 Wed.at Baltimore 7:05 ..."
"2","Cavaliers, Hokies Play Host","Akron at No. 12 Virginia &lt;br&gt;   Where:  Scott Stadium, Charlottesville&lt;br&gt;   When:  3 p.m.    Radio:  WTNT-570    Tickets:  Sold out &lt;br&gt;   Another Top QB:  Like North Carolina's Darian Durant last week, Akron quarterback Charlie Frye is the key to his team's slim chances of upsetting the Cavaliers. The 6-foot-4 senior, likened by Virginia Coach Al Groh to recent ACC stars Matt Schaub and Philip Rivers, ranks fourth among active NCAA quarterbacks with 9,048 career passing yards. In losses the past two weeks to Penn State and Middle Tennessee, Frye completed 60 of 80 passes for 622 yards. Weather permitting, he will present a challenge for an inexperienced Virginia secondary that made a few too many errors in deep coverage in its first two games."
"2","Boys' Top 20","Following is the Globe Top 20 in EMass for boys' soccer. No. Team Record Last 1. Stoneham 14-0-1 1 2. Framingham 10-0-5 2 3. Hingham 15-0-2 6 4. Weymouth 10-1-4 4 5. Medford 10-3-2 13 6. Woburn 10-2-2 10 7. Newton North 10-4-2 5 8. Everett 10-3-2 2 9. Lincoln-Sudbury 12-1-2 7 10. Dartmouth 14-1-0 9 11. BC High 7-1-6 ..."
"2","Girls' Top 20","Following is the Globe Top 20 in EMass for girls' soccer. No. Team Record Last 1. Belmont 14-0-1 1 2. Oliver Ames 13-1-0 2 3. Lynnfield 15-0-0 3 4. Norwell 15-0-0 5 5. Andover 14-1-0 4 6. Bishop Feehan 11-1-1 6 7. Old Rochester 14-1-1 7 8. Marshfield 10-1-4 9 9. Wellesley 9-0-5 8 10. Weymouth 12-1-2 10 11. Rockland ..."
"2","Girls Top 20","Following is the Globe Top 20 in EMass for girls' soccer. No. Team Record Last 1. Belmont 12-0-1 1 2. Oliver Ames 11-1-0 2 3. Lynnfield 12-0-0 3 4. Andover 12-0-0 4 5. Norwell 12-0-0 5 6. Bishop Feehan 11-1-1 9 7. Old Rochester 12-1-1 7 8. Wellesley 9-0-4 8 9. Marshfield 8-1-4 6 10. Weymouth 10-1-2 10 11. Rockland ..."
"2","Red Sox, Yanks Play for Pennant Tonight (AP)","AP - The Boston Red Sox and New York Yankees will play one game tonight for the American League pennant. The Red Sox pulled even in the series last night with a 4-2 win as Boston won its third straight after losing the first three. Derek Lowe (14-12) goes for Boston tonight against either Kevin Brown (10-6) or Javier Vazquez (14-10) for New York. This afternoon, Pete Munro (4-7) pitches for the Houston Astros against Matt Morris (15-10) for the St. Louis Cardinals. The Astros lead that series 3 games to 2."
"2","Area College Football Capsules","Navy at Tulsa &lt;br&gt;   Where:  Skelly Stadium    When:  7 p.m. &lt;br&gt;   Shooting for 3-0:  Navy is off to its first 2-0 start since 199
```

### Лабораторная №5

Модели SVM на вход подаются векторные представления документов из третьей лабораторной. Используются 4 класса, определенные в выбранном датасете. Для каждой из четырех ядерных функций используется свой файл с вычисленными метриками, отражающими насколько эффективной оказалась классификация.
При этом варьируется количество итераций. Пример файла  `assets/svm-models-evaluation/svm_rbf` с результатами для ядерной функции rbf:
```
Iterations	Training time (sec)	Accuracy    Error rate  Recall (micro)	Recall (macro)	Precision (micro)	Precision (macro)	F1 (micro)	F1 (macro)
100	5	0.6993421052631579	0.3006578947368421	0.3986842105263158	0.39263278846754757	0.3986842105263158	0.5954483496300247	0.3986842105263158	0.47322539999857577
500	30	0.7255263157894738	0.2744736842105263	0.45105263157894737	0.44616256492310613	0.45105263157894737	0.5864405891984817	0.45105263157894737	0.5067732679441409
1000	58	0.7506578947368421	0.24934210526315786	0.5013157894736842	0.49681192164357635	0.5013157894736842	0.6014152057448169	0.5013157894736842	0.5441319680060693
5000	214	0.7851973684210527	0.21480263157894738	0.5703947368421053	0.5666831357683508	0.5703947368421053	0.6352035934825412	0.5703947368421053	0.5989901634580246
10000	256	0.807078947368421	0.19292105263157897	0.6141578947368421	0.6110859188804241	0.6141578947368421	0.6587231305522924	0.6141578947368421	0.6340109636187725
20000	254	0.8216666666666667	0.17833333333333332	0.6433333333333333	0.6406877742884729	0.6433333333333333	0.6757410391410583	0.6433333333333333	0.657747715555969
30000	255	0.8320864661654135	0.16791353383458646	0.6641729323308271	0.6618319567227934	0.6641729323308271	0.6887574934232739	0.6641729323308271	0.6750263294749003
40000	256	0.8399013157894737	0.1600986842105263	0.6798026315789474	0.6776900935485337	0.6798026315789474	0.6990761091876598	0.6798026315789474	0.6882170013926586
50000	210	0.8946052631578948	0.10539473684210526	0.7892105263157895	0.7886970513287167	0.7892105263157895	0.7910015358310414	0.7892105263157895	0.7898476126741465
60000	210	0.8946052631578948	0.10539473684210526	0.7892105263157895	0.7886970513287167	0.7892105263157895	0.7910015358310414	0.7892105263157895	0.7898476126741465
80000	211	0.8946052631578948	0.10539473684210526	0.7892105263157895	0.7886970513287167	0.7892105263157895	0.7910015358310414	0.7892105263157895	0.7898476126741465
```


## Запуск тестов

Для проверки результатов работы системы используются модульные тесты. Для их запуска необходимо использовать следующее:
```
cd projects/nrogalenko
PYTHONPATH=source python -m unittest source/tests/*test.py
```
Указанная команда позволяет запустить все тесты. Для запуска тестов для конкретных лабораторных вместо `*test.py` необходимо написать:

- Лабораторная №1 - tokenization_test.py
- Лабораторная №2 - edit_distance_test.py
- Лабораторная №3 - vectorization_test.py
- Лабораторная №5 - evaluation_metrics_test.py



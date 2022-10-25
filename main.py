print("Hello World!")
number=1
boolean_True = True
boolean_False = False
boolean_Statement_T = 1 < 2
boolean_Statement_F = 1 > 2

print(boolean_Statement_T, boolean_Statement_F)
number = 12345
print(number)
print(365 , "days in a year.")


number = input()
print("number =", number)
name = input("Type your name: ")
print("Your name is", name, '!')

list = [1, "Workearly", 22]
list2 = [2, 6, list, 'Python']
print(list2)

tuple_example = (1, "Workearly", 22)

print(tuple_example)

string1 = "My name is"
string2 = "George!"
finalString = string1 + " " + string2
print(finalString)


# defining a function
def function_example(str):
    # This function will print the passed string
    print(str)

# calling a function function_example("Workearly")


from collections import Counter

mylist = ['a', 'b', 'c', 'a', 'a', 'a', 'b', 'c']
count = Counter(mylist)
print(Counter(mylist))

count.update(['a', 'b', 'b', 'c'])
print("Updated", count)

print("Count of \'b\':", count['b'])


def add(a, b):
    return a + 5, b + 5


sum = add(3, 2)
print(sum)

x, y, z, w = 2, 5, 1, 0
m=min(x,y,z,w)
print(m)

num = 1234
reversed_num = 0

while num != 0:
    digit = num % 10
    reversed_num = reversed_num * 10 + digit
    num //= 10

print("Reversed Number: " + str(reversed_num))

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(type(arr))

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)

import numpy as np

arr = np.array([[1, 2], [3, 4]])

print(arr)

import numpy as np

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(arr)

import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.shape)

import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
mean = df.mean(numeric_only=True)
print(mean)

median = df.median(numeric_only=True)
print(median)
max_v = df.max(numeric_only=True)
print(max_v)

summary = df.describe()
print(summary)
cn = df.groupby('category_name')
print(cn.first())
cn2 = df.groupby(['category_name', 'city'])
print(cn2.first())

import pandas as pd
import numpy as np
df = pd.read_csv("finance_liquor_sales.csv")
cn = df.groupby('category_name')
print(cn.aggregate(np.sum))
import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
cnc = df.groupby(['category_name', 'city'])
print(cnc.agg({'bottles_sold': 'sum', 'sale_dollars': 'mean'}))
import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
ng = df.groupby('vendor_name')
print(ng.filter(lambda x: len(x) >= 20))

import pandas as pd
d1 = {'Name': ['Mary', 'John', 'Alice', 'Bob'],
         'Age': [27, 24, 22, 32],
         'Position': ['Data Analyst', 'Trainee', 'QA Tester', 'IT']}
d2 = {'Name': ['Steve', 'Tom', 'Jenny', 'Nick'],
          'Age': [37, 25, 24, 52],
          'Position': ['IT', 'Data Analyst', 'Consultant', 'IT']}
df1 = pd.DataFrame(d1, index=[0, 1, 2, 3])
df2 = pd.DataFrame(d2, index=[4, 5, 6, 7])
result = pd.concat([df1, df2])
print(result)
import pandas as pd
d1 = {'key': ['a', 'b', 'c', 'd'],
         'Name': ['Mary', 'John', 'Alice', 'Bob']}
d2 = {'key': ['a', 'b', 'c', 'd'],
          'Age': [27, 24, 22, 32]}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
result = pd.merge(df1, df2, on='key')
print(result)
import pandas as pd
d1 = {'Name': ['Mary', 'John', 'Alice', 'Bob'],
         'Age': [27, 24, 42, 32]}
d2 = {'Position': ['Data Analyst', 'Trainee', 'QA Tester', 'IT'],
          'Years_of_experience':[5, 1, 10, 3] }
df1 = pd.DataFrame(d1, index=[0, 1, 2, 3])
df2 = pd.DataFrame(d2, index=[0, 2, 3, 4])
result = df1.join(df2, how='inner')
print(result)
import pandas as pd
L = [5, 10, 15, 20, 25]
ds = pd.Series(L)
print(ds)


import pandas as pd
d = {'col1': [1, 2, 3, 4, 7, 11],
       'col2': [4, 5, 6, 9, 5, 0],
       'col3': [7, 5, 8, 12, 1,11]}
ds = pd.DataFrame(d)
s1 = df1.iloc[:, 0]
print("1st column as a Series:")
print(s1)
print(type(s1))


import pandas as pd
L = [5, 10, 15, 20, 25]
ds = pd.Series(L)
print(ds)
import pandas as pd
d = {'col1': [1, 2, 3, 4, 7, 11],
       'col2': [4, 5, 6, 9, 5, 0],
       'col3': [7, 5, 8, 12, 1,11]}
ds = pd.DataFrame(d)
s1 = df1.iloc[:, 0]
print("1st column as a Series:")
print(s1)
print(type(s1))
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head(20))
import pandas as pd
df = pd.read_csv('data.csv')
for i, j in df.iterrows():
   print(i, j)

import pandas as pd
import numpy as np
data=pd.read_csv('1.supermarket.csv')

print (data.head())
print ('\nShape of dataset:', data.shape)

print (data.info())

print(data.columns)
x=data.groupby('item_name')
x=x.sum()
print(x.head(1))

import matplotlib.pyplot as plt

plt.plot([0, 10], [0, 300])

plt.show()

import matplotlib.pyplot as plt

plt.plot([0, 10], [0, 300], 'o')

plt.show()
import matplotlib.pyplot as plt

plt.plot([0, 2, 4, 6, 8, 10], [3, 8, 1, 10, 5, 12])

plt.show()
import matplotlib.pyplot as plt

plt.plot([0, 2, 4], [3, 8, 1], marker='o')

plt.show()

import matplotlib.pyplot as plt

plt.plot([0, 10], [0, 300])

plt.title("Title")
plt.xlabel("X - Axis")
plt.ylabel("y - Axis")

plt.show()
import matplotlib.pyplot as plt

plt.plot([0, 10], [0, 300])

plt.grid()

plt.show()

import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot([0, 2, 4, 6, 8, 10], [3, 8, 1, 10, 5, 12])

plt.subplot(2, 1, 1)
plt.plot([0, 10], [0, 300])

plt.show()
import matplotlib.pyplot as plt
import numpy as np

x = np.array([99, 86, 87, 88, 111, 86,
              103, 87, 94, 78, 77, 85, 86])

y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])

plt.scatter(x, y)

plt.show()
import matplotlib.pyplot as plt
import numpy as np

x = np.array([99, 86, 87, 88, 111, 86,
              103, 87, 94, 78, 77, 85, 86])
y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
plt.scatter(x, y)

x = np.array([100, 105, 84, 105, 90, 99,
              90, 95, 94, 100, 79, 112, 91, 80, 85])
y = np.array([2, 2, 8, 1, 15, 8, 12, 9,
              7, 3, 11, 4, 7, 14, 12])
plt.scatter(x, y)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])

y = np.array([4, 5, 1, 10])

plt.bar(x, y)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

mylabels = np.array(["Potatoes",
                     "Bacon", "Tomatoes", "Sausages"])

x = np.array([25, 35, 15, 25])

plt.pie(x, labels=mylabels)
plt.legend()

plt.show()
import matplotlib.pyplot as plt

age = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
cardiac_cases = [5, 15, 20, 40, 55, 55, 70, 80, 90, 95]
survival_chances = [99, 99, 90, 90, 80, 75, 60, 50, 30, 25]
plt.xlabel("Age")
plt.ylabel("Percentage")
plt.plot(age, cardiac_cases, color='black', linewidth=2,
label="Cardiac Cases",
marker='o',markerfacecolor='red',markersize=12)
plt.plot(age, survival_chances, color='yellow', linewidth=3,
label=" Survival_Chances ",
marker='o',markerfacecolor='green',markersize=12)
plt.legend(loc='lower right', ncol=1)

plt.show()
import numpy as np
import matplotlib.pyplot as plt

products = np.array([
    ["Apple", "Orange"],
    ["Beef", "Chicken"],
    ["Candy", "Chocolate"],
    ["Fish", "Bread"],
    ["Eggs", "Bacon"]])

random = np.random.randint(2, size=5)
choices = []

counter = 0
for product in products:
    choices.append(product[random[counter]])
    counter += 1

print(choices)
percentages = []

for i in range(4):
    percentages.append(np.random.randint(25))

percentages.append(100 - np.sum(percentages))

print(percentages)
plt.pie(percentages, labels=choices)
plt.legend(loc='lower right', ncol=1)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('1.supermarket.csv')
q=data.groupby('item_name').quantity.sum()
plt.bar(q.index, q, color=['orange','purple','yellow','red','green','blue','cyan'])
plt.xlabel('Items')
plt.xticks(rotation=6)
plt.ylabel('Number of Items Ordered')
plt.title ('Most ordered Supermarket\’s Items')
plt.show()

import requests

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
print(url_text)
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')
print(s.prettify())
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')
print(s.title)
print(s.title.string)
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')
tag = s.find_all('a')
print(tag)
import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')
tables = s.find_all('table')
print(tables)

import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')
my_table = s.find_all('table', class_='wikitable sortable plainrowheaders')
table_links = my_table.find
print(tables)

import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')
my_table = s.find_all('table', class_='wikitable sortable plainrowheaders')
table_links = my_table.find
actors = []
for links in table_links:
      actors.append(links.get('title'))
print(actors)


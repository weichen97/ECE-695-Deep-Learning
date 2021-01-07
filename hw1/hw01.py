import random
import string

random.seed(0)

for i in range(3):
	if i == 0:
		first = []
		for j in range(10):
			letters = ''
			for k in range(5):
				letters = letters + (random.choice(string.ascii_lowercase))
			first.append(letters)
	if i == 1:
		middle = []
		for j in range(10):
			letters = ''
			for k in range(5):
				letters = letters + (random.choice(string.ascii_lowercase))
			middle.append(letters)
	if i == 2:
		last = []
		for j in range(10):
			letters = ''
			for k in range(5):
				letters = letters + (random.choice(string.ascii_lowercase))
			last.append(letters)
#print(first,middle,last)

class People:
	def __init__(self, first, middle, last, order = 'first_name_first'):
		self.first_names = first
		self.middle_names = middle
		self.last_names = last
		self.flag = order
	
	def __iter__(self):
		self.num = 0
		return self
		
	def __next__(self):
		if self.num > 9:
			raise StopIteration
		else:
			self.num += 1
			#return [self.first_names[self.num-1], self.middle_names[self.num-1], self.last_names[self.num-1]]
			
			if self.flag == 'first_name_first':
				#return [self.first_names[self.num-1], self.middle_names[self.num-1], self.last_names[self.num-1]]
				self.t = 0
			elif self.flag == 'last_name_first':
				#return [self.last_names[self.num-1], self.middle_names[self.num-1], self.first_names[self.num-1]]
				self.t = 1
			elif self.flag == 'last_name_with_comma_first':
				#return [self.last_names[self.num-1]+',', self.middle_names[self.num-1], self.first_names[self.num-1]
				self.t = 2
			
			output = [[self.first_names[self.num-1], self.middle_names[self.num-1], self.last_names[self.num-1]],
			[self.last_names[self.num-1], self.middle_names[self.num-1], self.first_names[self.num-1]],
			[self.last_names[self.num-1]+',', self.middle_names[self.num-1], self.first_names[self.num-1]]]
			return output[self.t]
	
	def __call__(self):
		x = sorted(self.last_names)
		for i in range(len(x)):
			print(x[i])
		

xobj1 = People(first, middle, last, 'first_name_first')
myiter = iter(xobj1)
for x in myiter:
	print(x[0], end = ' ')
	print(x[1], end = ' ')
	print(x[2])
print()

xobj2 = People(first, middle, last, 'last_name_first')
myiter = iter(xobj2)
for x in myiter:
	print(x[0], end = ' ')
	print(x[1], end = ' ')
	print(x[2])
print()

xobj3 = People(first, middle, last, 'last_name_with_comma_first')
myiter = iter(xobj3)
for x in myiter:
	print(x[0], end = ' ')
	print(x[1], end = ' ')
	print(x[2])
print()

x = People(first,middle,last)
x()
print()

class PeopleWithMoney(People):
	def __init__(self, first, middle, last, money):
		People.__init__(self,first, middle, last)
		self.wealth = money
	
	def __next__(self):
		if self.num > 9:
			raise StopIteration
			
		else:
			self.num += 1			
			output = [self.first_names[self.num-1], self.middle_names[self.num-1], self.last_names[self.num-1], self.wealth[self.num-1]]
			
			return output
	
	def __call__(self):
		wealth_sorted = sorted(self.wealth)
		for i in range(10):
			index = self.wealth.index(wealth_sorted[i])
			print(self.first_names[index], end = ' ')
			print(self.middle_names[index], end = ' ')
			print(self.last_names[index], end = ' ')
			print(self.wealth[index])
			
		
money = []
for i in range(10):
	money.append(random.randrange(0,1000,1))
#print(money)

yobj = PeopleWithMoney(first, middle, last, money)
y_iter = iter(yobj)
for y in y_iter:
	print(y[0], end = ' ')
	print(y[1], end = ' ')
	print(y[2], end = ' ')
	print(y[3])
print()
yobj()
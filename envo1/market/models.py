from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField
from datetime import datetime

# Create your models here.

class Product(models.Model):
	productId = models.AutoField(primary_key=True)
	productName = models.CharField(max_length=255)
	productIngredients = models.TextField()
	productCategory=models.CharField(max_length=255)
	productPrice = models.FloatField()
	productAbout = models.TextField()
	envScore = models.FloatField()
	imageUrl = models.CharField(max_length=255,default="")
    
# class ProductBought(models.Field):
# 	productId = models.IntegerField(blank=True,null=True)
# 	productName=models.CharField(max_length=255,null=True,default='')
# 	productPrice=models.FloatField(blank=True, null=True)
# 	productUserDist=models.IntegerField(blank=True,null=True)
# 	envScore=models.FloatField(blank=True,null=True)
	
# 	def db_type(self, connection):
# 	    return 'Text'

# 	def rel_db_type(self, connection):
# 	    return 'integer UNSIGNED'

# 	def to_python(self, value):
# 	    return json.loads(value)

# 	def get_prep_value(self, value):
# 	    return json.dumps(value)
	# def __init__(id:int,name1:str,price:float,quantity:int,envscore:float,distance:float):
	# 	self.id=id
	# 	self.name=name1
	# 	self.price=price
	# 	self.quantity=quantity
	# 	self.envscore=envscore
	# 	self.datetime=datetime.now()
	# 	self.regularisedenvscore=envscore-0.01*distance

class AppUser(models.Model):
	user = models.OneToOneField(User, on_delete = models.CASCADE)
	userId = models.AutoField(primary_key = True)
	productsBought = JSONField(default="{}")
	avgEnvScore=models.FloatField()



class envoFriendly(models.Model):
	meanenvscore=models.FloatField()

# class ProductBought(models.Model):
# 	userBoughtId = models.ForeignKey(AppUser, on_delete=models.PROTECT)
# 	productBoughtId = models.ForeignKey(Product, on_delete=models.PROTECT)

# 	def __unicode__(self):
# 		return str(self.userBoughtId)




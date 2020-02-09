import graphene
from django.db.models import Q
from graphene import ObjectType
from graphene_django import DjangoObjectType

from .models import Product


class ProductType(DjangoObjectType):
    class Meta:
        model = Product


class Query(ObjectType):
    products = graphene.List(ProductType,
                             search=graphene.String(),
                             searchtype=graphene.String())

    def resolve_products(self, info, search=None, searchtype=None, **kwargs):
        if search and searchtype == None:
            filter = (Q(productName__icontains=search))
            return Product.objects.filter(filter)
        elif search and searchtype == "category":
            filter = (Q(productCategory__icontains=search))
            return Product.objects.filter(filter)
        elif search and searchtype == "name":
            filter = (Q(productName__icontains=search))
            return Product.objects.filter(filter)
        elif search and searchtype == "full":
            filter = (Q(productName__icontains=search)
                      | Q(productAbout__icontains=search))
            return Product.objects.filter(filter)
        else:
            return Product.objects.all()

    # productId = models.AutoField(primary_key=True)


# productName = models.CharField(max_length=255)
# productIngredients = models.TextField()
# productCategory = models.CharField(max_length=255)
# productPrice = models.FloatField()
# productAbout = models.TextField()
# envScore = models.FloatField()
# imageUrl = models.CharField(max_length=255, default="")

# integrate DL solution here as predict after creation of product to compute the envscore


class AddProduct(graphene.Mutation):
    id = graphene.Int()
    name = graphene.String()
    ing = graphene.String()
    category = graphene.String()
    price = graphene.Float()
    about = graphene.String()
    envScore = graphene.Float()
    imageurl = graphene.String()

    class Arguments:
        name = graphene.String()
        ing = graphene.String()
        category = graphene.String()
        price = graphene.Float()
        about = graphene.String()
        # envScore=graphene.Float()
        imageurl = graphene.String()

    def mutate(self, info, name: str, ing: str, category: str, price: float,
               about: str, imageurl: str):
        # Add predict method here

        envscored = 3.141562789  # shere
        prod = Product(productName=name,
                       productIngredients=ing,
                       productCategory=category,
                       productPrice=price,
                       productAbout=about,
                       imageUrl=imageurl,
                       envScore=envscored)
        prod.save()

        return AddProduct(id=prod.productId,
                          name=prod.productName,
                          ing=prod.productIngredients,
                          category=prod.productCategory,
                          price=prod.productPrice,
                          about=prod.productAbout,
                          envScore=prod.envScore,
                          imageurl=prod.imageUrl)


class Mutation(graphene.ObjectType):
    add_product = AddProduct.Field()

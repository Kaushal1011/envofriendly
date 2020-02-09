import graphene
from django.db.models import Q
from graphene import ObjectType
from graphene_django import DjangoObjectType

from .models import AppUser


class AppUserType(DjangoObjectType):
    class Meta:
        model = AppUser


class Query(ObjectType):
    # users = graphene.List(UserType)
    # products = graphene.List(ProductType)
    app_users = graphene.List(AppUserType)

    def resolve_app_users(self, info):
        return AppUser.objects.all()


class addInstance(graphene.Mutation):

    # user = graphene.Field(UserType)
    # products = graphene.List(ProductType)
    app_users = graphene.List(AppUserType)

    class Arguments:
        userid = graphene.Int()
        # productsBought = graphene.String()
        envScore = graphene.Float()

    def mutate(self, info, userid, envScore):

        # temp_user_instance = User(id=userId, email="manavdarji@test.com", username="test")
        # temp_product_instance = Product()
        # inst = AppUser(user=user, userId=userId, productsBought=productsBought, envScore=envScore)
        # inst.save()

        curr_user = AppUser.objects.get(userId=userid)
        curr_user.avgEnvScore += envScore
        curr_user.save()
        # curr_user.productsBought

        return None


class Mutation(ObjectType):
    create_new_instance = addInstance.Field()

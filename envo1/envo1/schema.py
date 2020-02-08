import graphene
import market.schema
import market.schema_user
import graphql_jwt

class Query(market.schema.Query,market.schema_user.Query,graphene.ObjectType):
    pass

class Mutation(market.schema.Mutation,market.schema_user.Mutation,graphene.ObjectType):
    token_auth = graphql_jwt.ObtainJSONWebToken.Field()
    verify_token = graphql_jwt.Verify.Field()
    refresh_token = graphql_jwt.Refresh.Field()

schema=graphene.Schema(query=Query,mutation=Mutation)
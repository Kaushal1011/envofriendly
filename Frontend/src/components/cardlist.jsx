import React from 'react'
import Card from './card'
import { Query } from 'react-apollo'
import gql from 'graphql-tag'

class Cardlist extends React.Component {
    render() {
        let CAT_QUERY = gql`
        {
            products(search:"${this.props.category}",searchtype:"category"){
              productId,
              productName,
              productPrice,
              productCategory,
              imageUrl,
              productIngredients,
              productAbout,
              envScore
            }
          }`;
        return (<Query query={CAT_QUERY}>
            {({ loading, error, data }) => {
                if (loading) return <div>Fetching</div>
                if (error) return <div>Error</div>

                const products = data.products

                return (
                    <div className="cardlist">
                        {products.map(product => <Card product={product} />)}
                    </div>
                )
            }}
        </Query>)

    }
}

export default Cardlist;
import React from 'react'
import {FaBars} from 'react-icons/fa'
import { Link } from 'react-router-dom'
import { withRouter } from 'react-router'
import '../styles/sidebar.css'
import Score from './yourscore'
class Sidebar extends React.Component{
    
    render(){
        return(
            <>
            <FaBars className="menu-icon"/>
            <div className="sidebar" >

                <Score />
                <span className="cat">Categories</span>
            {
                this.props.menu.map((menu_item,index) => (
                        <Link key={index} to={menu_item.name} className="link-menu-items">{menu_item.name}</Link> 
                ))
            }
            
        </div>
        </>)
    }
}

export default withRouter(Sidebar);
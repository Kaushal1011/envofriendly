import React from 'react'
import '../styles/score.css'
class Score extends React.Component{

    render(){ 
        let color = 'black';

    if (this.props.value < 5) {
        color = 'red';
    }
    else if (this.props.value < 8 && this.props.value >= 5) {
        color = 'blue';
    } else {
        color = 'green';
    }
     return (
        <div className="score">
            <h3>Your Score</h3>
            <span className={"num "+color}>7</span>
            <span className="small">+ Just added 0.01</span>
        </div>);
    }
}
export default Score;
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class BondDataGenerator:
    def __init__(self, n_trades=1000, seed=42):
        """
        Initialize the bond data generator
        
        Parameters:
        n_trades (int): Number of bond trades to generate
        seed (int): Random seed for reproducibility
        """
        self.n_trades = n_trades
        np.random.seed(seed)
        random.seed(seed)
        
        # Credit rating agencies
        self.rating_agencies = ['Moody\'s', 'S&P', 'Fitch']
        
        # Credit rating categories with numerical mappings
        self.credit_ratings = {
            'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
            'A+': 17, 'A': 16, 'A-': 15,
            'BBB+': 14, 'BBB': 13, 'BBB-': 12,
            'BB+': 11, 'BB': 10, 'BB-': 9,
            'B+': 8, 'B': 7, 'B-': 6,
            'CCC+': 5, 'CCC': 4, 'CCC-': 3,
            'CC': 2, 'C': 1, 'D': 0
        }
        
        # Investment grade ratings
        self.investment_grade = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
        
    def generate_cusip(self):
        """Generate a realistic 9-digit CUSIP identifier"""
        # CUSIP format: 6 alphanumeric + 2 alphanumeric + 1 check digit
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        cusip = ''.join(random.choices(chars, k=8))
        
        # Simple check digit calculation (simplified)
        check_digit = random.choice('0123456789')
        return cusip + check_digit
    
    def generate_credit_rating(self, agency):
        """Generate credit rating based on agency and market conditions"""
        # Different agencies have slightly different rating distributions
        if agency == 'Moody\'s':
            # Moody's tends to be more conservative
            weights = [0.05, 0.08, 0.12, 0.15, 0.20, 0.15, 0.10, 0.08, 0.04, 0.03]
        elif agency == 'S&P':
            # S&P has more balanced distribution
            weights = [0.06, 0.10, 0.14, 0.16, 0.18, 0.14, 0.10, 0.06, 0.04, 0.02]
        else:  # Fitch
            # Fitch similar to S&P
            weights = [0.05, 0.09, 0.13, 0.15, 0.19, 0.15, 0.11, 0.07, 0.03, 0.02]
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        rating_categories = list(self.credit_ratings.keys())[:len(weights)]
        return np.random.choice(rating_categories, p=weights)
    
    def calculate_trade_rating(self, row):
        """
        Calculate trade rating (1-10) based on bond characteristics
        Creates a realistic normal distribution centered around 5-6
        """
        # Start with a base score around 5
        base_score = 5.0
        
        # Duration impact (shorter duration = lower risk = higher rating)
        # Scale duration impact to be smaller
        duration_factor = max(-1, min(1, (10 - row['duration']) / 15))
        
        # Yield impact (moderate yield = good, too high = risky)
        # Create a bell curve effect where moderate yields are best
        optimal_yield = 5.0
        yield_deviation = abs(row['yield'] - optimal_yield)
        yield_factor = max(-1, min(1, (3 - yield_deviation) / 3))
        
        # Credit rating impact (higher rating = lower risk = higher rating)
        credit_score = self.credit_ratings[row['moodys_rating']]
        # Scale credit impact to be moderate
        credit_factor = (credit_score / 21 - 0.5) * 2  # Range: -1 to 1
        
        # Price impact (closer to par = better, but with realistic bounds)
        price_deviation = abs(100 - row['price']) / 100
        price_factor = max(-1, min(1, (1 - price_deviation) * 2))
        
        # YTM vs YTW spread (smaller spread = better)
        ytm_ytw_spread = abs(row['ytm'] - row['ytw'])
        spread_factor = max(-0.5, min(0.5, (1 - ytm_ytw_spread) / 2))
        
        # Coupon-yield spread impact (positive spread = better, but not too much)
        coupon_yield_spread = row['base_coupon'] - row['yield']
        spread_impact = max(-0.5, min(0.5, coupon_yield_spread / 4))
        
        # Calculate final rating with moderate factors
        final_rating = (base_score + 
                       duration_factor + 
                       yield_factor + 
                       credit_factor + 
                       price_factor + 
                       spread_factor + 
                       spread_impact)
        
        # Add significant randomness to create normal distribution
        # Use a larger standard deviation to spread out the ratings
        final_rating += np.random.normal(0, 2.0)
        
        # Ensure rating is between 1 and 10
        final_rating = max(1, min(10, final_rating))
        
        return round(final_rating, 1)
    
    def generate_bond_data(self):
        """Generate comprehensive bond trading dataset"""
        
        data = []
        
        for i in range(self.n_trades):
            # Generate CUSIP
            cusip = self.generate_cusip()
            
            # Generate duration (1-30 years)
            duration = np.random.lognormal(2.5, 0.5)
            duration = max(1, min(30, duration))
            
            # Generate base coupon (0-15%)
            base_coupon = np.random.normal(5, 2)
            base_coupon = max(0, min(15, base_coupon))
            
            # Generate yield (0-20%)
            yield_rate = base_coupon + np.random.normal(2, 1.5)
            yield_rate = max(0, min(20, yield_rate))
            
            # Generate price (50-150, with some relationship to coupon and yield)
            price = 100 + (base_coupon - yield_rate) * 5 + np.random.normal(0, 10)
            price = max(50, min(150, price))
            
            # Generate YTM (yield to maturity)
            ytm = yield_rate + np.random.normal(0, 0.5)
            ytm = max(0, min(25, ytm))
            
            # Generate YTW (yield to worst)
            ytw = ytm + np.random.normal(0.2, 0.3)
            ytw = max(ytm, min(25, ytw))
            
            # Generate credit ratings for all three agencies
            moodys_rating = self.generate_credit_rating('Moody\'s')
            sp_rating = self.generate_credit_rating('S&P')
            fitch_rating = self.generate_credit_rating('Fitch')
            
            # Create row data
            row_data = {
                'cusip': cusip,
                'duration': round(duration, 2),
                'yield': round(yield_rate, 2),
                'base_coupon': round(base_coupon, 2),
                'price': round(price, 2),
                'ytm': round(ytm, 2),
                'ytw': round(ytw, 2),
                'moodys_rating': moodys_rating,
                'sp_rating': sp_rating,
                'fitch_rating': fitch_rating
            }
            
            data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate trade rating based on characteristics
        df['trade_rating'] = df.apply(self.calculate_trade_rating, axis=1)
        
        # Add some additional derived features
        df['price_to_par'] = df['price'] / 100
        df['coupon_yield_spread'] = df['base_coupon'] - df['yield']
        df['ytm_ytw_spread'] = df['ytw'] - df['ytm']
        
        # Add numerical credit ratings for analysis
        df['moodys_numeric'] = df['moodys_rating'].map(self.credit_ratings)
        df['sp_numeric'] = df['sp_rating'].map(self.credit_ratings)
        df['fitch_numeric'] = df['fitch_rating'].map(self.credit_ratings)
        
        # Add investment grade flag
        df['is_investment_grade'] = df['moodys_rating'].isin(self.investment_grade)
        
        return df
    
    def save_data(self, df, filename='bond_trading_data.csv'):
        """Save the generated data to CSV"""
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        print(f"Dataset shape: {df.shape}")
        return filename

if __name__ == "__main__":
    # Generate sample data
    generator = BondDataGenerator(n_trades=1000)
    bond_data = generator.generate_bond_data()
    
    # Display sample and statistics
    print("Sample of generated bond trading data:")
    print(bond_data.head())
    print("\nDataset statistics:")
    print(bond_data.describe())
    
    # Save data
    generator.save_data(bond_data) 
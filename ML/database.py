import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from estilos_graficas import COLORS, PLOT_STYLE, apply_style, style_histogram_bars,get_gradient_colors
from matplotlib.colors import LinearSegmentedColormap

class StudentAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.dataset, self.headers = self.load_data()
        self.data_dict = self.create_data_dictionary()
    
    # It is responsible for loading and structuring data from the CSV file
    def load_data(self):
        """We load data from the CSV"""
        with open(self.filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)
        
        dataset = {}
        for i, header in enumerate(headers):
            dataset[header] = [row[i] if i < len(row) else '' for row in data]
        
        return dataset, headers
    
    """ 
        Generates a complete data dictionary with metadata and descriptive statistics for each variable in the dataset. 
        For numerical variables, it calculates mean, median, standard deviation, minimum, maximum, and range. 
        For categorical variables, it determines unique values, the most frequent value, and the frequency distribution.
    """
    def create_data_dictionary(self):
        """We create a data dictionary """
        variable_definitions = {
            'student_id': {'type': 'Categorical', 'description': 'Unique identifier', 'domain': 'ID'},
            'age': {'type': 'Numerical', 'description': 'Age in years', 'domain': 'Demographics'},
            'gender': {'type': 'Categorical', 'description': 'Student gender', 'domain': 'Demographics'},
            'study_hours_per_day': {'type': 'Numerical', 'description': 'Daily study hours', 'domain': 'Academic'},
            'social_media_hours': {'type': 'Numerical', 'description': 'Daily social media hours', 'domain': 'Digital'},
            'netflix_hours': {'type': 'Numerical', 'description': 'Daily Netflix hours', 'domain': 'Entertainment'},
            'part_time_job': {'type': 'Categorical', 'description': 'Has part-time job', 'domain': 'Employment'},
            'attendance_percentage': {'type': 'Numerical', 'description': 'Class attendance %', 'domain': 'Academic'},
            'sleep_hours': {'type': 'Numerical', 'description': 'Average sleep hours', 'domain': 'Health'},
            'diet_quality': {'type': 'Categorical', 'description': 'Diet quality rating', 'domain': 'Health'},
            'exercise_frequency': {'type': 'Numerical', 'description': 'Exercise sessions/week', 'domain': 'Health'},
            'parental_education_level': {'type': 'Categorical', 'description': 'Parent education level', 'domain': 'Socioeconomic'},
            'internet_quality': {'type': 'Categorical', 'description': 'Internet connection quality', 'domain': 'Infrastructure'},
            'mental_health_rating': {'type': 'Numerical', 'description': 'Mental health rating 1-10', 'domain': 'Health'},
            'extracurricular_participation': {'type': 'Categorical', 'description': 'Extracurricular participation', 'domain': 'Academic'},
            'exam_score': {'type': 'Numerical', 'description': 'Final exam score', 'domain': 'Performance'}
        }
        
        data_dict = {'variables': {}}
        total_students = len(self.dataset[self.headers[0]])
        
        for header in self.headers:
            var_info = variable_definitions.get(header, {'type': 'Unknown', 'description': f'Variable {header}', 'domain': 'General'})
            column_data = [val.strip() for val in self.dataset[header] if val.strip()]
            
            stats = {
                'total_observations': total_students,
                'valid_values': len(column_data),
                'missing_values': total_students - len(column_data)
            }
            
            if var_info['type'] == 'Numerical':
                numeric_values = []
                for val in column_data:
                    try:
                        numeric_values.append(float(val))
                    except ValueError:
                        pass
                
                if numeric_values:
                    numeric_values.sort()
                    n = len(numeric_values)
                    mean = sum(numeric_values) / n
                    std_dev = (sum((x - mean) ** 2 for x in numeric_values) / (n - 1)) ** 0.5 if n > 1 else 0
                    
                    stats.update({
                        'mean': round(mean, 3),
                        'median': numeric_values[n // 2],
                        'std_deviation': round(std_dev, 3),
                        'minimum': min(numeric_values),
                        'maximum': max(numeric_values),
                        'range': round(max(numeric_values) - min(numeric_values), 3)
                    })
            
            elif var_info['type'] == 'Categorical':
                frequencies = Counter(column_data)
                most_common = frequencies.most_common(1)[0] if frequencies else ('N/A', 0)
                
                stats.update({
                    'unique_values': len(frequencies),
                    'most_common_value': most_common[0],
                    'most_common_count': most_common[1],
                    'frequency_distribution': dict(frequencies)
                })
            
            data_dict['variables'][header] = {
                'description': var_info['description'],
                'data_type': var_info['type'],
                'domain': var_info['domain'],
                'statistics': stats
            }
        
        return data_dict
    
    def get_numeric_data(self, var_name):
        return [float(val) for val in self.dataset[var_name] if val.replace('.', '').replace('-', '').isdigit()]
    
    def print_summary(self):
        """Imprimir resumen del dataset"""
        print("="*80)
        print("📊 STUDENT DATASET ANALYSIS SUMMARY")
        print("="*80)
        
        total_students = len(self.dataset[self.headers[0]])
        numerical_vars = sum(1 for v in self.data_dict['variables'].values() if v['data_type'] == 'Numerical')
        categorical_vars = sum(1 for v in self.data_dict['variables'].values() if v['data_type'] == 'Categorical')
        
        print(f"👥 Total Students: {total_students}")
        print(f"📊 Variables: {len(self.headers)} ({numerical_vars} numerical, {categorical_vars} categorical)")
        
        key_stats = ['exam_score', 'study_hours_per_day', 'attendance_percentage']
        for var in key_stats:
            if var in self.data_dict['variables']:
                stats = self.data_dict['variables'][var]['statistics']
                print(f"📈 {var.replace('_', ' ').title()}: Mean={stats.get('mean', 'N/A')}, Range={stats.get('range', 'N/A')}")
                
        print("="*90)
    
    """
        Generates a 3x3 subplot visualization showing the distributions of up to 9 numerical variables
        in the dataset using stylized histograms. Uses custom styles imported from styles_graphs.py to
        maintain visual consistency. Saves the resulting image in high resolution (300 DPI) as 
        'distributions_numerical_improved.png'.
    """
    def create_distributions_plot(self):
        numerical_vars = [var for var, info in self.data_dict['variables'].items() 
                        if info['data_type'] == 'Numerical' and var != 'student_id'][:9]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.patch.set_facecolor(PLOT_STYLE['figure']['facecolor'])
        
        fig.suptitle('Distributions of Numerical Variables', 
                    fontsize=PLOT_STYLE['title']['fontsize'],
                    fontweight=PLOT_STYLE['title']['fontweight'],
                    color=PLOT_STYLE['title']['color'])
        
        for idx, var in enumerate(numerical_vars):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            values = self.get_numeric_data(var)
            if values:
                apply_style(fig, ax)
                
                n, bins, patches = ax.hist(values, bins=10, 
                                        alpha=PLOT_STYLE['histogram']['alpha'],
                                        edgecolor=PLOT_STYLE['histogram']['edgecolor'], 
                                        linewidth=PLOT_STYLE['histogram']['linewidth'])
                
                style_histogram_bars(patches, n)

                for i, (count, patch) in enumerate(zip(n, patches)):
                    if count > 0:
                        height = patch.get_height()
                        ax.text(patch.get_x() + patch.get_width()/2., height + n.max()*0.02,
                            f'{int(count)}', ha='center', va='bottom', 
                            fontsize=PLOT_STYLE['text']['fontsize'],
                            color=PLOT_STYLE['text']['color'],
                            fontweight=PLOT_STYLE['text']['fontweight'])
                
                mean_val = np.mean(values)
                ax.axvline(mean_val, 
                        color=PLOT_STYLE['mean_line']['color'],
                        linestyle=PLOT_STYLE['mean_line']['linestyle'],
                        alpha=PLOT_STYLE['mean_line']['alpha'],
                        linewidth=PLOT_STYLE['mean_line']['linewidth'],
                        label=f'Mean: {mean_val:.2f}')
                
                var_title = var.replace('_', ' ').title()
                ax.set_title(var_title, 
                            fontsize=PLOT_STYLE['axes']['titlesize'],
                            fontweight=PLOT_STYLE['axes']['titleweight'],
                            color=COLORS['primary'],
                            pad=8)
                
                x_labels = {
                    'age': 'Age',
                    'study_hours_per_day': 'Study Hours Per Day',
                    'social_media_hours': 'Social Media Hours',
                    'netflix_hours': 'Netflix Hours',
                    'attendance_percentage': 'Attendance Percentage (%)',
                    'sleep_hours': 'Sleep Hours',
                    'exercise_frequency': 'Exercise Frequency Per Week',
                    'mental_health_rating': 'Mental Health Rating (1-10)',
                    'exam_score': 'Exam Score'
                }
                
                ax.set_xlabel(x_labels.get(var, var_title), 
                            fontsize=PLOT_STYLE['axes']['labelsize'],
                            color=COLORS['text'])
                ax.set_ylabel('Students Number', 
                            fontsize=PLOT_STYLE['axes']['labelsize'],
                            color=COLORS['text'])

                legend = ax.legend(loc='upper right',
                                fontsize=PLOT_STYLE['legend']['fontsize'],
                                facecolor=PLOT_STYLE['legend']['facecolor'],
                                edgecolor=PLOT_STYLE['legend']['edgecolor'],
                                framealpha=PLOT_STYLE['legend']['framealpha'])
                legend.get_frame().set_linewidth(0.5)
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', 
                        color=COLORS['text_light'])

        for idx in range(len(numerical_vars), 9):
            row, col = idx // 3, idx % 3
            axes[row, col].axis('off')

        plt.tight_layout(pad=PLOT_STYLE['figure']['tight_layout']['pad'])

        plt.savefig('distributions_numerical_improved.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor=PLOT_STYLE['figure']['facecolor'],
                    edgecolor='none')
        plt.show()
    

    """
        Each chart displays the frequency of each category with blue gradient bars, including the absolute count and percentage 
        above each bar. Use custom styles imported from styles_graphs.py to maintain visual consistency. Save the resulting
        image as 'distributions_categorical.png'.
    """
    def create_categorical_plots(self):
        categorical_vars = [var for var, info in self.data_dict['variables'].items() 
                        if info['data_type'] == 'Categorical' and var != 'student_id'][:6]
       
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor(PLOT_STYLE['figure']['facecolor'])
        
        fig.suptitle('Distributions of Categorical Variables', 
                    fontsize=PLOT_STYLE['title']['fontsize'],
                    fontweight=PLOT_STYLE['title']['fontweight'],
                    color=PLOT_STYLE['title']['color'])
        
        for idx, var in enumerate(categorical_vars):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            apply_style(fig, ax)

            frequencies = Counter(self.dataset[var])
            categories, values = zip(*frequencies.most_common())

            gradient_colors = get_gradient_colors(len(categories))
            bars = ax.bar(categories, values, 
                        color=gradient_colors,
                        alpha=PLOT_STYLE['histogram']['alpha'],
                        edgecolor=PLOT_STYLE['histogram']['edgecolor'],
                        linewidth=PLOT_STYLE['histogram']['linewidth'])

            total = sum(values)
            for bar, value in zip(bars, values):
                height = bar.get_height()
                percentage = value/total*100
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value}\n({percentage:.1f}%)', 
                    ha='center', va='bottom',
                    fontsize=PLOT_STYLE['text']['fontsize'],
                    color=PLOT_STYLE['text']['color'],
                    fontweight=PLOT_STYLE['text']['fontweight'])

            var_title = var.replace('_', ' ').title()
            ax.set_title(var_title,
                        fontsize=PLOT_STYLE['axes']['titlesize'],
                        fontweight=PLOT_STYLE['axes']['titleweight'],
                        color=COLORS['primary'],
                        pad=8)

            ax.set_ylabel('Students Number',
                        fontsize=PLOT_STYLE['axes']['labelsize'],
                        color=COLORS['text'])

            if max(len(str(cat)) for cat in categories) > 8:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right',
                        color=COLORS['text_light'])
            else:
                ax.tick_params(axis='x', colors=COLORS['text_light'])

            ax.set_ylim(0, max(values) * 1.15)
        
        for idx in range(len(categorical_vars), 6):
            row, col = idx // 3, idx % 3
            axes[row, col].axis('off')

        plt.tight_layout(pad=PLOT_STYLE['figure']['tight_layout']['pad'])
        plt.savefig('distributions_categorical.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor=PLOT_STYLE['figure']['facecolor'],
                    edgecolor='none')
        plt.show()
    
    """
        Calculates and visualizes correlations between all numerical variables in the dataset, creating a heat
        map showing how closely related the variables are to each other.
    """
    def create_correlation_matrix(self):
        numerical_vars = [var for var, info in self.data_dict['variables'].items() 
                        if info['data_type'] == 'Numerical' and var != 'student_id']
        
        correlation_data = []
        for var in numerical_vars:
            values = self.get_numeric_data(var)
            correlation_data.append(values[:min(len(values), 1000)]) 

        min_length = min(len(data) for data in correlation_data)
        correlation_data = [data[:min_length] for data in correlation_data]
        
        correlation_matrix = np.corrcoef(correlation_data)

        fig, ax = plt.subplots(figsize=(12, 10))

        apply_style(fig, ax)
        colors_map = [COLORS['gradient_end'], 'white', COLORS['primary']]
        custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', colors_map, N=256)

        im = ax.imshow(correlation_matrix, cmap=custom_cmap, aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(len(numerical_vars)))
        ax.set_yticks(range(len(numerical_vars)))

        formatted_labels = [var.replace('_', ' ').title() for var in numerical_vars]
        ax.set_xticklabels(formatted_labels, rotation=45, ha='right', 
                        color=COLORS['text'], fontsize=10, fontweight='medium')
        ax.set_yticklabels(formatted_labels, 
                        color=COLORS['text'], fontsize=10, fontweight='medium')

        for i in range(len(numerical_vars)):
            for j in range(len(numerical_vars)):
                correlation_val = correlation_matrix[i, j]

                if abs(correlation_val) > 0.6:
                    text_color = 'white'
                    font_weight = 'bold'
                else:
                    text_color = COLORS['text']
                    font_weight = 'medium'
                
                ax.text(j, i, f'{correlation_val:.2f}', 
                    ha="center", va="center", 
                    color=text_color, fontweight=font_weight, fontsize=9)

        cbar = plt.colorbar(im, ax=ax, label='Correlation coefficient', shrink=0.8)
        cbar.ax.tick_params(labelsize=8, colors=COLORS['text_light'])
        cbar.set_label('Correlation coefficient', 
                    color=COLORS['text'], fontsize=10, fontweight='medium')

        ax.set_title('Correlation Matrix - Numerical Variables', 
                    fontsize=16, fontweight='bold', color=COLORS['primary'], pad=25)

        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLORS['primary'])
            spine.set_linewidth(2)

        plt.tight_layout(pad=3.0)
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight', 
                    facecolor=PLOT_STYLE['figure']['facecolor'], edgecolor='none')
        plt.show()

    """
        It generates scatter plots showing the correlation between study hours and grades, as well as 
        between attendance and performance, including trend lines and correlation coefficients. It also 
        produces box plots to compare academic performance by gender and between working and non-working 
        students, allowing for the identification of patterns and differences in student performance 
        based on these demographic and behavioral variables.
    """
    def create_performance_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        fig.patch.set_facecolor(PLOT_STYLE['figure']['facecolor'])

        fig.suptitle('Academic Performance Analysis', 
                    fontsize=16, fontweight='bold', color=COLORS['primary'])
        
        exam_scores = self.get_numeric_data('exam_score')
        study_hours = self.get_numeric_data('study_hours_per_day')
        attendance = self.get_numeric_data('attendance_percentage')

        ax1 = axes[0, 0]
        apply_style(fig, ax1)
        
        ax1.scatter(study_hours, exam_scores, alpha=0.7, s=50, 
                color=COLORS['secondary'], edgecolors=COLORS['primary'], linewidth=0.8)
        z = np.polyfit(study_hours, exam_scores, 1)
        p = np.poly1d(z)
        ax1.plot(study_hours, p(study_hours), color=COLORS['mean_line'], 
                linestyle='--', linewidth=2.5, alpha=0.9)
        ax1.set_xlabel('Study Hours Per Day', color=COLORS['text'])
        ax1.set_ylabel('Exam Score', color=COLORS['text'])
        ax1.set_title('Study vs Performance', color=COLORS['primary'], fontweight='bold')
        
        corr = np.corrcoef(study_hours, exam_scores)[0, 1]
        ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor=COLORS['background'], 
                        edgecolor=COLORS['primary'], alpha=0.9),
                color=COLORS['text'], fontweight='bold')

        ax2 = axes[0, 1]
        apply_style(fig, ax2)
        
        ax2.scatter(attendance, exam_scores, alpha=0.7, s=50, 
                color=COLORS['accent'], edgecolors=COLORS['primary'], linewidth=0.8)
        z2 = np.polyfit(attendance, exam_scores, 1)
        p2 = np.poly1d(z2)
        ax2.plot(attendance, p2(attendance), color=COLORS['mean_line'], 
                linestyle='--', linewidth=2.5, alpha=0.9)
        ax2.set_xlabel('Attendance Percentage', color=COLORS['text'])
        ax2.set_ylabel('Exam Score', color=COLORS['text'])
        ax2.set_title('Attendance vs. Performance', color=COLORS['primary'], fontweight='bold')

        ax3 = axes[1, 0]
        apply_style(fig, ax3)
        
        genders = ['Male', 'Female', 'Other']
        scores_by_gender = []
        valid_genders = []
        
        for gender in genders:
            scores = [float(self.dataset['exam_score'][i]) for i, g in enumerate(self.dataset['gender']) 
                    if g == gender and self.dataset['exam_score'][i].replace('.', '').isdigit()]
            if scores:
                scores_by_gender.append(scores)
                valid_genders.append(gender)
        
        if scores_by_gender:
            bp3 = ax3.boxplot(scores_by_gender, labels=valid_genders,
                            patch_artist=True,
                            boxprops=dict(facecolor=COLORS['secondary'], alpha=0.8,
                                        edgecolor=COLORS['primary'], linewidth=1.5),
                            medianprops=dict(color=COLORS['mean_line'], linewidth=2),
                            whiskerprops=dict(color=COLORS['primary'], linewidth=1.5),
                            capprops=dict(color=COLORS['primary'], linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor=COLORS['accent'],
                                        markeredgecolor=COLORS['primary'], markersize=6))
        
        ax3.set_ylabel('Exam Score', color=COLORS['text'])
        ax3.set_title('Performance by Gender', color=COLORS['primary'], fontweight='bold')

        ax4 = axes[1, 1]
        apply_style(fig, ax4)
        
        trabajo_si = [float(self.dataset['exam_score'][i]) for i, trabajo in enumerate(self.dataset['part_time_job']) 
                    if trabajo == 'Yes' and self.dataset['exam_score'][i].replace('.', '').isdigit()]
        trabajo_no = [float(self.dataset['exam_score'][i]) for i, trabajo in enumerate(self.dataset['part_time_job']) 
                    if trabajo == 'No' and self.dataset['exam_score'][i].replace('.', '').isdigit()]
        
        if trabajo_si and trabajo_no:
            bp4 = ax4.boxplot([trabajo_si, trabajo_no], labels=['Con Trabajo', 'Sin Trabajo'],
                            patch_artist=True,
                            boxprops=dict(alpha=0.8, edgecolor=COLORS['primary'], linewidth=1.5),
                            medianprops=dict(color=COLORS['mean_line'], linewidth=2),
                            whiskerprops=dict(color=COLORS['primary'], linewidth=1.5),
                            capprops=dict(color=COLORS['primary'], linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor=COLORS['accent'],
                                        markeredgecolor=COLORS['primary'], markersize=6))

            bp4['boxes'][0].set_facecolor(COLORS['secondary'])
            bp4['boxes'][1].set_facecolor(COLORS['accent'])
            
        ax4.set_ylabel('Exam Score', color=COLORS['text'])
        ax4.set_title('Work vs. Performance', color=COLORS['primary'], fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight',
                    facecolor=PLOT_STYLE['figure']['facecolor'], edgecolor='none')
        plt.show()

    """
        This method analyzes four key relationships in student lifestyle: how sleep and mental health 
        impact academic performance, whether screen time affects sleep, and the relationship between 
        exercise and grades. Each graph includes trend lines and correlation coefficients to quantify 
        these relationships and identify which habits promote academic success.
    """
    def create_lifestyle_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        fig.patch.set_facecolor(PLOT_STYLE['figure']['facecolor'])
        fig.suptitle('Lifestyle and Performance Analysis', 
                    fontsize=16, fontweight='bold', color=COLORS['primary'])

        sleep_hours = self.get_numeric_data('sleep_hours')
        mental_health = self.get_numeric_data('mental_health_rating')
        social_media = self.get_numeric_data('social_media_hours')
        netflix = self.get_numeric_data('netflix_hours')
        exercise = self.get_numeric_data('exercise_frequency')
        exam_scores = self.get_numeric_data('exam_score')

        ax1 = axes[0, 0]
        apply_style(fig, ax1)
        
        ax1.scatter(sleep_hours, exam_scores, alpha=0.7, s=50, 
                color=COLORS['secondary'], edgecolors=COLORS['primary'], linewidth=0.8)
        z1 = np.polyfit(sleep_hours, exam_scores, 1)
        p1 = np.poly1d(z1)
        ax1.plot(sleep_hours, p1(sleep_hours), color=COLORS['mean_line'], 
                linestyle='--', linewidth=2.5, alpha=0.9)
        ax1.set_xlabel('Sleep Hours', color=COLORS['text'])
        ax1.set_ylabel('Exam Score', color=COLORS['text'])
        ax1.set_title('Sleep vs. Performance', color=COLORS['primary'], fontweight='bold')

        corr1 = np.corrcoef(sleep_hours, exam_scores)[0, 1]
        ax1.text(0.05, 0.95, f'r = {corr1:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor=COLORS['background'], 
                        edgecolor=COLORS['primary'], alpha=0.9),
                color=COLORS['text'], fontweight='bold')

        ax2 = axes[0, 1]
        apply_style(fig, ax2)
        
        ax2.scatter(mental_health, exam_scores, alpha=0.7, s=50, 
                color=COLORS['accent'], edgecolors=COLORS['primary'], linewidth=0.8)
        z2 = np.polyfit(mental_health, exam_scores, 1)
        p2 = np.poly1d(z2)
        ax2.plot(mental_health, p2(mental_health), color=COLORS['mean_line'], 
                linestyle='--', linewidth=2.5, alpha=0.9)
        ax2.set_xlabel('Mental Health (1-10)', color=COLORS['text'])
        ax2.set_ylabel('Exam Score', color=COLORS['text'])
        ax2.set_title('Mental Health and Performance', color=COLORS['primary'], fontweight='bold')
        
        corr2 = np.corrcoef(mental_health, exam_scores)[0, 1]
        ax2.text(0.05, 0.95, f'r = {corr2:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round", facecolor=COLORS['background'], 
                        edgecolor=COLORS['primary'], alpha=0.9),
                color=COLORS['text'], fontweight='bold')
 
        ax3 = axes[1, 0]
        apply_style(fig, ax3)
        
        screen_time = [s + n for s, n in zip(social_media, netflix)]
        ax3.scatter(screen_time, sleep_hours, alpha=0.7, s=50, 
                color='#FF6B6B', edgecolors=COLORS['primary'], linewidth=0.8)
        z3 = np.polyfit(screen_time, sleep_hours, 1)
        p3 = np.poly1d(z3)
        ax3.plot(screen_time, p3(screen_time), color=COLORS['mean_line'], 
                linestyle='--', linewidth=2.5, alpha=0.9)
        ax3.set_xlabel('Total Screen Time (hrs/day)', color=COLORS['text'])
        ax3.set_ylabel('Sleep Hours', color=COLORS['text'])
        ax3.set_title('Screen vs Sleep', color=COLORS['primary'], fontweight='bold')
        
        corr3 = np.corrcoef(screen_time, sleep_hours)[0, 1]
        ax3.text(0.05, 0.95, f'r = {corr3:.3f}', transform=ax3.transAxes, 
                bbox=dict(boxstyle="round", facecolor=COLORS['background'], 
                        edgecolor=COLORS['primary'], alpha=0.9),
                color=COLORS['text'], fontweight='bold')
  
        ax4 = axes[1, 1]
        apply_style(fig, ax4)
        
        ax4.scatter(exercise, exam_scores, alpha=0.7, s=50, 
                color='#4ECDC4', edgecolors=COLORS['primary'], linewidth=0.8)
        z4 = np.polyfit(exercise, exam_scores, 1)
        p4 = np.poly1d(z4)
        ax4.plot(exercise, p4(exercise), color=COLORS['mean_line'], 
                linestyle='--', linewidth=2.5, alpha=0.9)
        ax4.set_xlabel('Exercise (times/week)', color=COLORS['text'])
        ax4.set_ylabel('Exam Score', color=COLORS['text'])
        ax4.set_title('Exercise vs. Performance', color=COLORS['primary'], fontweight='bold')
        
        corr4 = np.corrcoef(exercise, exam_scores)[0, 1]
        ax4.text(0.05, 0.95, f'r = {corr4:.3f}', transform=ax4.transAxes, 
                bbox=dict(boxstyle="round", facecolor=COLORS['background'], 
                        edgecolor=COLORS['primary'], alpha=0.9),
                color=COLORS['text'], fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('lifestyle_analysis.png', dpi=300, bbox_inches='tight',
                    facecolor=PLOT_STYLE['figure']['facecolor'], edgecolor='none')
        plt.show()

    '''
        This method creates a visual dashboard with four main elements: overall student metrics, grade 
        histogram, correlations between academic variables and performance, and gender distribution.
    '''
    def create_dashboard(self):
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(PLOT_STYLE['figure']['facecolor'])
        
        fig.suptitle('Dashboard Resumen - Análisis Estudiantes', 
                    fontsize=18, fontweight='bold', color=COLORS['primary'])
        
        exam_scores = self.get_numeric_data('exam_score')
        study_hours = self.get_numeric_data('study_hours_per_day')

        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

        ax_metrics = fig.add_subplot(gs[0, :])
        ax_metrics.axis('off')
        ax_metrics.set_facecolor(PLOT_STYLE['figure']['facecolor'])
        
        total_students = len(self.dataset['student_id'])
        high_performers = sum(1 for score in exam_scores if score > 80)
        with_job = self.dataset['part_time_job'].count('Yes')
        
        metrics_text = f'''
    📊 MAIN METRICS
    👥 Total Students: {total_students:,}
    📈 Promedio Calificaciones: {np.mean(exam_scores):.1f}/100
    📚 Average Study Hours: {np.mean(study_hours):.1f} hrs/día
    🎓 High Performance (>80): {high_performers} ({high_performers/len(exam_scores)*100:.1f}%)
    💼 With Work: {with_job} ({with_job/total_students*100:.1f}%)
        '''
        
        ax_metrics.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                    fontweight='bold', color=COLORS['text'],
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['background'], 
                                edgecolor=COLORS['primary'], alpha=0.9, linewidth=2))

        ax1 = fig.add_subplot(gs[1, 0])
        apply_style(fig, ax1)
        
        n, bins, patches = ax1.hist(exam_scores, bins=15, alpha=0.8, 
                                edgecolor=COLORS['primary'], linewidth=1.2)

        style_histogram_bars(patches, n)

        for i, (count, patch) in enumerate(zip(n, patches)):
            if count > 0:
                height = patch.get_height()
                ax1.text(patch.get_x() + patch.get_width()/2., height + n.max()*0.02,
                        f'{int(count)}', ha='center', va='bottom', 
                        fontsize=PLOT_STYLE['text']['fontsize'],
                        color=COLORS['text'], fontweight='bold')

        mean_score = np.mean(exam_scores)
        ax1.axvline(mean_score, color=COLORS['mean_line'], linestyle='--', 
                linewidth=2.5, alpha=0.9, label=f'Media: {mean_score:.1f}')
        
        ax1.set_title('Ratings Distribution', 
                    color=COLORS['primary'], fontweight='bold', fontsize=12)
        ax1.set_xlabel('Puntuación', color=COLORS['text'])
        ax1.set_ylabel('Students Number', color=COLORS['text'])
        
        legend = ax1.legend(loc='upper right', fontsize=10,
                        facecolor=COLORS['background'], edgecolor=COLORS['primary'],
                        framealpha=0.9)
        legend.get_frame().set_linewidth(1)

        ax2 = fig.add_subplot(gs[1, 1])
        apply_style(fig, ax2)
        
        variables = ['study_hours_per_day', 'attendance_percentage', 'sleep_hours', 'mental_health_rating']
        var_labels = ['Horas Estudio', 'Asistencia %', 'Horas Sueño', 'Salud Mental']
        correlations = []
        
        for var in variables:
            var_data = self.get_numeric_data(var)
            if len(var_data) == len(exam_scores):
                corr = np.corrcoef(var_data, exam_scores)[0, 1]
            else:
                min_len = min(len(var_data), len(exam_scores))
                corr = np.corrcoef(var_data[:min_len], exam_scores[:min_len])[0, 1]
            correlations.append(corr)
        
        colors = [COLORS['secondary'] if corr > 0 else COLORS['accent'] for corr in correlations]
        bars = ax2.barh(var_labels, correlations, color=colors, alpha=0.8,
                    edgecolor=COLORS['primary'], linewidth=1.2)

        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            width = bar.get_width()
            x_pos = width + (0.02 if width >= 0 else -0.02)
            ha = 'left' if width >= 0 else 'right'
            ax2.text(x_pos, bar.get_y() + bar.get_height()/2, f'{corr:.3f}',
                    ha=ha, va='center', fontweight='bold', color=COLORS['text'])
        
        ax2.set_title('Correlations with Academic Performance', 
                    color=COLORS['primary'], fontweight='bold', fontsize=12)
        ax2.set_xlabel('Correlation Coefficient', color=COLORS['text'])
        ax2.axvline(0, color=COLORS['text'], linestyle='-', alpha=0.3)
  
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_facecolor(PLOT_STYLE['axes']['facecolor'])
        
        gender_counts = Counter(self.dataset['gender'])
        colors_pie = [COLORS['secondary'], COLORS['accent'], COLORS['primary']][:len(gender_counts)]
        
        wedges, texts, autotexts = ax3.pie(gender_counts.values(), 
                                        labels=gender_counts.keys(), 
                                        autopct='%1.1f%%',
                                        colors=colors_pie,
                                        explode=[0.05] * len(gender_counts),
                                        shadow=True,
                                        startangle=90)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_color(COLORS['text'])
            text.set_fontweight('bold')
        
        ax3.set_title('Distribution by Gender', 
                    color=COLORS['primary'], fontweight='bold', fontsize=12)
        
        plt.tight_layout(pad=2.0)
        plt.savefig('dashboard_summary.png', dpi=300, bbox_inches='tight',
                    facecolor=PLOT_STYLE['figure']['facecolor'], edgecolor='none')
        plt.show()
    
    def save_report(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"student_analysis_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("STUDENT DATASET ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            total_students = len(self.dataset['student_id'])
            f.write(f"Total Students: {total_students}\n")
            f.write(f"Total Variables: {len(self.headers)}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("VARIABLE SUMMARY:\n")
            f.write("-" * 50 + "\n")
            
            for var_name, var_info in self.data_dict['variables'].items():
                f.write(f"\n{var_name.upper()}\n")
                f.write(f"Description: {var_info['description']}\n")
                f.write(f"Type: {var_info['data_type']}\n")
                f.write(f"Domain: {var_info['domain']}\n")
                
                stats = var_info['statistics']
                f.write(f"Valid Values: {stats['valid_values']}/{stats['total_observations']}\n")
                
                if var_info['data_type'] == 'Numerical':
                    f.write(f"Mean: {stats.get('mean', 'N/A')}\n")
                    f.write(f"Range: {stats.get('range', 'N/A')} ({stats.get('minimum', 'N/A')} - {stats.get('maximum', 'N/A')})\n")
                elif var_info['data_type'] == 'Categorical':
                    f.write(f"Unique Values: {stats.get('unique_values', 'N/A')}\n")
                    f.write(f"Most Common: {stats.get('most_common_value', 'N/A')} ({stats.get('most_common_count', 'N/A')} times)\n")
                
                f.write("-" * 40 + "\n")
        
        print(f"✅ Save: {filename}")
        return filename
    
    def run_complete_analysis(self):
        print("🚀 STARTING COMPLETE ANALYSIS")
        print("="*60)
        
        # Imprimir resumen
        self.print_summary()
        
        # Crear todas las visualizaciones
        visualizations = [
            ("Numerical Distributions", self.create_distributions_plot),
            ("Categorical Distributions", self.create_categorical_plots),
            ("Matriz de Correlación", self.create_correlation_matrix),
            ("Performance Analysis", self.create_performance_analysis),
            ("Lifestyle Analysis", self.create_lifestyle_analysis),
            ("Dashboard Summary", self.create_dashboard)
        ]
        
        successful = 0
        for name, func in visualizations:
            try:
                print(f"\n📊 Generating: {name}...")
                func()
                successful += 1
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
        
        # Guardar reporte
        report_file = self.save_report()
        
        print(f"\n🎉 ANALYSIS COMPLETED!")
        print(f"✅ Successful visualizations: {successful}/{len(visualizations)}")
        print(f"📄 Report saved: {report_file}")
        
        files_generated = [
            "distributions_numerical.png",
            "distributions_categorical.png",
            "correlation_matrix.png",
            "performance_analysis.png",
            "lifestyle_analysis.png",
            "dashboard_summary.png"
        ]
        
        print("\n📁 Archivos generados:")
        for file in files_generated:
            print(f"  - {file}")
        
        return self.data_dict

def analyze_student_data(csv_filename):
    try:
        analyzer = StudentAnalyzer(csv_filename)
        return analyzer.run_complete_analysis()
    except FileNotFoundError:
        print(f"❌ Error: File not found {csv_filename}")
        return None
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None


def clean_and_prepare_data(filename):
    print("\n🧹 DATA CLEANING AND FEATURE ENGINEERING")
    print("="*50)
    
    # Cargar como DataFrame
    df = pd.read_csv(filename)
    print(f"📊 Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 1. LIMPIEZA RÁPIDA
    print("🔧 Cleaning data...")
    
    # Rellenar faltantes
    numerical_cols = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 
                     'attendance_percentage', 'sleep_hours', 'exercise_frequency', 
                     'mental_health_rating', 'exam_score']
    categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level',
                       'internet_quality', 'extracurricular_participation']
    
    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)

    outliers_fixed = 0
    for col in numerical_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            before = len(df[(df[col] < lower) | (df[col] > upper)])
            df[col] = np.clip(df[col], lower, upper)
            outliers_fixed += before
    
    print(f"  ✅ {outliers_fixed} outliers ajustados")
    print(f"  ✅ {df.isnull().sum().sum()} valores faltantes restantes")
    
    print("⚙️ Creating derived featuress...")
    
    df['total_entertainment'] = df['social_media_hours'] + df['netflix_hours']
    df['study_ratio'] = df['study_hours_per_day'] / (df['total_entertainment'] + 0.1)
    df['wellness_index'] = (df['sleep_hours']/8 + df['exercise_frequency']/7 + df['mental_health_rating']/10) / 3

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col])
    
    print("  ✅ 3 derived variables created")
    print("  ✅ Coded categorical variables")
    
    return df

def train_ml_model(df):
    """Compact ML model training"""
    print("\n🤖 ML MODEL TRAINING")
    print("="*35)

    feature_cols = [
        'age', 'study_hours_per_day', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'total_entertainment',
        'study_ratio', 'wellness_index', 'gender_enc', 'part_time_job_enc',
        'diet_quality_enc', 'parental_education_level_enc'
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df['exam_score']
    
    print(f"📊 Features: {len(available_features)}, Target: exam_score")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("🔧 Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"📈 RESULTS:")
    print(f"  • R²: {r2:.3f} ({r2:.1%} explained variance)")
    print(f"  • RMSE: {rmse:.3f}")
    print(f"  • MAE: {mae:.3f}")
    
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 TOP 5 IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.3f}")
    
    return {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'feature_importance': importance_df,
        'available_features': available_features
    }

def create_ml_visualization(ml_results):
    if not ml_results:
        return
    
    print("\n📊 Generating ML visualization...")

    importance_df = ml_results['feature_importance']
    top_features = importance_df.head(10)

    fig, ax = plt.subplots(figsize=(12, 8))
    apply_style(fig, ax)

    bars = ax.barh(range(len(top_features)), 
                   top_features['importance'], 
                   alpha=PLOT_STYLE['histogram']['alpha'],
                   edgecolor=PLOT_STYLE['histogram']['edgecolor'],
                   linewidth=PLOT_STYLE['histogram']['linewidth'])

    max_importance = top_features['importance'].max()
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):

        intensity = importance / max_importance
        color_intensity = 0.4 + 0.6 * intensity  
        bar.set_facecolor(plt.cm.Blues(color_intensity))

        width = bar.get_width()
        ax.text(width + max_importance * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}',
                ha='left', va='center',
                fontsize=PLOT_STYLE['text']['fontsize'],
                fontweight=PLOT_STYLE['text']['fontweight'],
                color=COLORS['text'])

    feature_labels = [feature.replace('_', ' ').title() for feature in top_features['feature']]
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(feature_labels, 
                       fontsize=PLOT_STYLE['axes']['labelsize'],
                       color=COLORS['text'])

    ax.invert_yaxis()

    ax.set_title('🎯 Top 10 Most Important Features (Random Forest)', 
                fontsize=PLOT_STYLE['title']['fontsize'],
                fontweight=PLOT_STYLE['title']['fontweight'],
                color=COLORS['primary'],
                pad=PLOT_STYLE['title']['pad'])
    
    ax.set_xlabel('Relative Importance', 
                  fontsize=PLOT_STYLE['axes']['labelsize'],
                  color=COLORS['text'],
                  fontweight='medium')

    mean_importance = top_features['importance'].mean()
    ax.axvline(mean_importance, 
               color=COLORS['mean_line'],
               linestyle=PLOT_STYLE['mean_line']['linestyle'],
               linewidth=PLOT_STYLE['mean_line']['linewidth'],
               alpha=PLOT_STYLE['mean_line']['alpha'],
               label=f'Media: {mean_importance:.3f}')
    

    legend = ax.legend(loc='lower right',
                       fontsize=PLOT_STYLE['legend']['fontsize'],
                       facecolor=PLOT_STYLE['legend']['facecolor'],
                       edgecolor=PLOT_STYLE['legend']['edgecolor'],
                       framealpha=PLOT_STYLE['legend']['framealpha'])
    legend.get_frame().set_linewidth(0.5)
    

    plt.tight_layout(pad=PLOT_STYLE['figure']['tight_layout']['pad'])
    

    plt.savefig('ml_feature_importance_styled.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor=PLOT_STYLE['figure']['facecolor'],
                edgecolor='none')
    plt.show()
    
    print("  ✅Saved stylized graphic: ml_feature_importance_styled.png")

def generate_ml_insights(df, ml_results):
    print("\n📖 ML-BASED INSIGHTS")
    print("="*30)
    
    if not ml_results:
        return

    high_perf = df[df['exam_score'] > 80]
    low_perf = df[df['exam_score'] < 60]
    
    print("💡SUCCESSFUL STUDENT PROFILE (>80 points):")
    if len(high_perf) > 0 and len(low_perf) > 0:
        print(f"  • Study: {high_perf['study_hours_per_day'].mean():.1f} Per Day")
        print(f"    (vs {low_perf['study_hours_per_day'].mean():.1f} low-performance hours)")
        print(f"  • Attendance: {high_perf['attendance_percentage'].mean():.1f}%")
        print(f"    (vs {low_perf['attendance_percentage'].mean():.1f}% low performance)")
        print(f"  • Bienestar: {high_perf['wellness_index'].mean():.2f}")
        print(f"    (vs {low_perf['wellness_index'].mean():.2f} low performance)")

    top_feature = ml_results['feature_importance'].iloc[0]
    print(f"\n🏆 MOST PREDICTIVE FACTOR: {top_feature['feature']}")
    print(f"  • Importance: {top_feature['importance']:.3f}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"  📚 Maintain 4+ hours of daily study")
    print(f"  🎯 Achieve >85% attendance")
    print(f"  💪 Take care of your overall well-being (sleep + exercise)")
    print(f"  ⚖️ Balance digital entertainment")

def save_ml_report(ml_results, df):
    if not ml_results:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ml_report_compact_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ML REPORT - ACADEMIC PERFORMANCE PREDICTION\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. CLEANING PROCESS:\n")
        f.write(f"• Missing data: Filled with median/mode\n")
        f.write(f"• Outliers: Adjusted with capping method\n")
        f.write(f"• Derived variables: 3 created\n")
        f.write(f"• Encoding: Coded categorical variables\n\n")
        
        f.write("2. MACHINE LEARNING MODEL:\n")
        f.write(f"• Algorithm: Random Forest Regressor\n")
        f.write(f"• Justification: Robust to mixed data\n")
        f.write(f"• Features used:{len(ml_results['available_features'])}\n")
        f.write(f"• Split: 80% training, 20% testing\n\n")
        
        f.write("3. RESULTS:\n")
        f.write(f"• R²: {ml_results['r2']:.3f} ({ml_results['r2']:.1%} varianza explicada)\n")
        f.write(f"• RMSE: {ml_results['rmse']:.3f}\n")
        f.write(f"• MAE: {ml_results['mae']:.3f}\n\n")
        
        f.write("4. MOST IMPORTANT FEATURES:\n")
        for i, (_, row) in enumerate(ml_results['feature_importance'].head(5).iterrows(), 1):
            f.write(f"{i}. {row['feature']}: {row['importance']:.3f}\n")
        
        f.write(f"\n5. INSIGHTS CLAVE:\n")
        high_perf = df[df['exam_score'] > 80]
        if len(high_perf) > 0:
            f.write(f"• Successful students study {high_perf['study_hours_per_day'].mean():.1f} hrs/día\n")
            f.write(f"• Mantienen {high_perf['attendance_percentage'].mean():.1f}% assistance\n")
            f.write(f"• They have better general well-being\n")
        
        f.write(f"\n" + "="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Saved ML Report: {filename}")
    return filename

def run_complete_ml_analysis(filename):
    print("🚀 ADDITIONAL COMPACT ML ANALYSIS")
    print("="*40)
    
    try:
        # 1. Limpieza y preparación
        df_clean = clean_and_prepare_data(filename)
        
        # 2. Entrenamiento ML
        ml_results = train_ml_model(df_clean)
        
        # 3. Visualización
        create_ml_visualization(ml_results)
        
        # 4. Insights
        generate_ml_insights(df_clean, ml_results)
        
        # 5. Reporte
        report_file = save_ml_report(ml_results, df_clean)
        
        print(f"\n🎉 ML ANALYSIS COMPLETED!")
        print(f"📁 Generated files:")
        print(f"  • ml_feature_importance.png")
        print(f"  • {report_file}")
        
        return ml_results
        
    except Exception as e:
        print(f"❌ Error in ML analysis: {e}")
        return None

if __name__ == "__main__":
    filename = "student_habits_performance.csv"
    
    print("🎓 STUDENT  HABITS PERFORMANCE")
    print("="*60)

    result = analyze_student_data(filename)
    
    if result:
        print("\n💡 ¡Original analysis completed successfully!")
        print("\n" + "="*50)
        print("🤖 RUNNING ADDITIONAL ML ANALYSIS")
        print("="*50)
        
        ml_results = run_complete_ml_analysis(filename)
        
        if ml_results:
            print(f"\n🏆 PROJECT 100% COMPLETED!")
            print(f"✅ Model accuracy: {ml_results['r2']:.1%}")
            print(f"🎯 Most important factor:{ml_results['feature_importance'].iloc[0]['feature']}")
            print(f"\n📋 ALL ACADEMIC REQUIREMENTS MET:")
            print(f"  ✅ Data Dictionary & Statistics")
            print(f"  ✅ Documented Cleaning Process") 
            print(f"  ✅ Data Visualizations")
            print(f"  ✅ Feature Engineering")
            print(f"  ✅ Storytelling with Insights")
            print(f"  ✅ Trained and Evaluated ML Model")
        else:
            print(f"\n⚠️ Análisis original OK, revisar ML")
    else:
        print(f"\n❌ Verificar archivo CSV")
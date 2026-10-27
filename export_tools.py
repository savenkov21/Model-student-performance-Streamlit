from io import BytesIO
import zipfile,json
import pandas as pd
import streamlit as st


def install_figure_capture(page):
    st.session_state['_figure_page']=page
    st.session_state['_figure_counter']=0
    if getattr(st.pyplot,'_captured',False):return
    original=st.pyplot
    def pyplot(fig=None,*args,**kwargs):
        if fig is not None:
            import matplotlib.pyplot as plt
            n=st.session_state.get('_figure_counter',0)+1;st.session_state['_figure_counter']=n
            key=f"stage_{st.session_state.get('_figure_page',0)}_{n:02d}"
            outputs={}
            for fmt in ['png','pdf']:
                buf=BytesIO();fig.savefig(buf,format=fmt,dpi=200,bbox_inches='tight');outputs[fmt]=buf.getvalue()
            st.session_state.setdefault('figure_exports',{})[key]=outputs
        result=original(fig,*args,**kwargs)
        if fig is not None:
            plt.close(fig)
        return result
    pyplot._captured=True;st.pyplot=pyplot


def all_exports_zip():
    bio=BytesIO()
    with zipfile.ZipFile(bio,'w',zipfile.ZIP_DEFLATED) as z:
        for name,outputs in st.session_state.get('figure_exports',{}).items():
            for fmt,data in outputs.items():z.writestr(f'figures/{name}.{fmt}',data)
        for key in ['df','X_train_raw','X_test_raw','X_train','X_test','meta_X_train','meta_X_test']:
            table=st.session_state.get(key)
            if isinstance(table,pd.DataFrame):z.writestr(f'tables/{key}.csv',table.to_csv(index=True,index_label='row_id'))
        if 'X_train_raw' in st.session_state:
            from data_protocol import groups_for, cv_folds
            rows=[]
            for part in ['train','test']:
                x=st.session_state[f'X_{part}_raw']
                rows.extend({'row_id':i,'group_id':g,'part':part} for i,g in zip(x.index,groups_for(x)))
            assignment=pd.DataFrame(rows)
            assignment['fold']=-1
            x=st.session_state.X_train_raw
            folds=cv_folds(x,st.session_state.y_train_cls,int(st.session_state.get('meta_generation_params',{}).get('cv_folds',5)),int(st.session_state.get('random_state',42)))
            for k,(_,valid) in enumerate(folds):assignment.loc[assignment.row_id.isin(x.index[valid]),'fold']=k
            z.writestr('tables/split_assignment.csv',assignment.to_csv(index=False))
        for strategy,run in st.session_state.get('evaluation_runs',{}).items():
            z.writestr(f'{strategy}/metrics.csv',run['table'].to_csv(index=False))
            z.writestr(f'{strategy}/metrics.tex',run['table'].to_latex(index=False,float_format='%.4f'))
            pred=pd.DataFrame({'row_id':run['test_index'],'target':run['y_cont'],'true_class':run['y_cls']})
            for name,p in run['predictions'].items():pred[name]=p
            for name,p in run.get('continuous',{}).items():pred[f'{name}_grade']=p
            for name,p in run.get('probabilities',{}).items():
                for c in range(p.shape[1]):pred[f'{name}_P{c}']=p[:,c]
            z.writestr(f'{strategy}/test_predictions.csv',pred.to_csv(index=False))
        for key,table in st.session_state.get('statistical_tables',{}).items():
            z.writestr(f'statistics/{key}.csv',table.to_csv(index=False));z.writestr(f'statistics/{key}.tex',table.to_latex(index=False,float_format='%.4f'))
        z.writestr('experiment.json',json.dumps({k:st.session_state.get(k) for k in ['target_col','strategy_type','Q1','Q3','test_size','random_state','preprocessing_config','first_level_params','meta_generation_params','second_level_params','strategy_snapshots']},ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))
    return bio.getvalue()


def render_figure_downloads(page):
    outputs={k:v for k,v in st.session_state.get('figure_exports',{}).items() if k.startswith(f'stage_{page}_')}
    if outputs:
        with st.expander('Завантаження графіків цієї сторінки (PNG / PDF)',expanded=False):
            for name,files in outputs.items():
                st.caption(name)
                a,b=st.columns(2)
                a.download_button('PNG',files['png'],name+'.png','image/png',key=name+'png',on_click='ignore')
                b.download_button('PDF',files['pdf'],name+'.pdf','application/pdf',key=name+'pdf',on_click='ignore')

from .predict import Yolo_test
import os,shutil,glob,json

def load_json(fp):
	with open(fp,'r') as f:
		return json.load(f)
def dump_json(obj,fp):
	with open(fp,'w') as f:
		json.dump(obj,f)

class Detector(Yolo_test):
	def load_dict(self,fp):
		return load_json(fp)
	def save_dict(self,obj,fp):
		return dump_json(obj,fp)

	def detect_dir_to_dir(self,src_dir,dst_dir,coors_file=None,verbose=True):
		shutil.rmtree(dst_dir) if os.path.exists(dst_dir) else None
		fs=glob.glob(src_dir+'/*.jpg')
		dic={}
		for j,f in enumerate(fs):
			imgs,coors=self.predict_from_file(f)
			name=os.path.basename(f).split('.')[0]
			d=dst_dir+'/'+name
			os.makedirs(d) if not os.path.exists(d) else None
			for i,img in enumerate(imgs):
				f2=d+'/'+name+'_'+str(i)+'.jpg'
				self.save_img(img,f2)
			if verbose:
				print('%s , detected %s , %s objects found .'%(j,f,len(imgs)))
			dic[f]=coors

		if coors_file:
			self.save_dict(dic,coors_file)
			print('save coordinates dict to %s'%(coors_file))



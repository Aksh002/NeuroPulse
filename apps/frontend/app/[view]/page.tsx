import Dashboard from '../../components/dashboard';
import {notFound} from 'next/navigation';
export default async function Page({params}:{params:Promise<{view:string}>}) {
  const {view}=await params;
  if(view!=='machines'&&view!=='replay'&&view!=='models'&&view!=='experiments')notFound();
  return <Dashboard view={view}/>;
}

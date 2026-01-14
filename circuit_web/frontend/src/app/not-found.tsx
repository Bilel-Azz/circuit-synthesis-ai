import Link from 'next/link';

export default function NotFound() {
    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-background text-foreground text-center p-4">
            <div className="glass-card p-12 rounded-2xl border border-border max-w-md w-full">
                <h2 className="text-6xl font-black text-primary mb-4">404</h2>
                <p className="text-xl font-bold mb-2">Page Not Found</p>
                <p className="text-muted-foreground mb-8">Could not find requested resource</p>
                <Link
                    href="/"
                    className="inline-block px-6 py-3 rounded-lg bg-primary text-primary-foreground font-bold hover:bg-primary/90 transition-all shadow-lg hover:shadow-primary/25"
                >
                    Return Home
                </Link>
            </div>
        </div>
    );
}
